#ifndef COAL_PTR
#define COAL_PTR

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <cooperative_groups/scan.h>
#include <cuda.h>
#include <cuda_runtime_api.h>

#include <gallatin/allocators/alloc_utils.cuh>
#include <gallatin/data_structs/ds_utils.cuh>

#include "assert.h"
#include "stdio.h"

namespace cg = cooperative_groups;

#define CACHE_PRINT 0

// helper_macro
// define macros
#define MAX_VALUE(nbits) ((1ULL << (nbits)) - 1)
#define BITMASK(nbits) ((nbits) == 64 ? 0xffffffffffffffff : MAX_VALUE(nbits))

#define SET_BIT_MASK(index) ((1ULL << index))

// a pointer list managing a set section of device memory


namespace gpu_pointers {


   template <typename T>
   struct coalesce_pointer {

      using my_type = coalesce_pointer<T>;

      T * internal_reference;

      //generate device reference
      static __host__ my_type * generate_on_device(T original_value){

         T * host_internal = gallatin::utils::get_host_version<T>();

         host_internal[0] = original_value;

         T * dev_internal = gallatin::utils::move_to_device<T>(host_internal);

         my_type * host_version = gallatin::utils::get_host_version<my_type>();

         host_version->internal_reference = dev_internal;

         return gallatin::utils::move_to_device<my_type>(host_version);

      }

      static __host__ void free_on_device(my_type * device_version){

         my_type * host_version = gallatin::utils::move_to_host<my_type>(device_version);

         //cudaFree(host_version->internal_reference);

         cudaFreeHost(host_version);

      }

      __device__ T atomicAdd(T new_obj){


         // auto coalesced_team = cg::coalesced_threads();

         // //auto labeled_team = cg::labeled_partition(coalesced_team, (uint64_t) this);



         // // if (coalesced_team.size() != labeled_team.size()){
         // //    printf("Label is needed for correctness\n");
         // // }

         // //group_lower_value is everyone below my
         // T group_lower_value = cg::exclusive_scan(coalesced_team, new_obj, cg::plus<T>());

         // T existing_value;

         // if (coalesced_team.thread_rank() == coalesced_team.size()-1){

         //    existing_value = gallatin::utils::typed_atomic_add(internal_reference, new_obj+group_lower_value);

         // }

         // existing_value = coalesced_team.shfl(existing_value, coalesced_team.size()-1);

         // return existing_value+group_lower_value;

         return gallatin::utils::typed_atomic_add(internal_reference, new_obj);

      }

      __device__ T atomicExch(T new_obj){


         auto coalesced_team = cg::coalesced_threads();

         //auto labeled_team = cg::labeled_partition(coalesced_team, (uint64_t) this);

         T prev_val = coalesced_team.shfl(new_obj, (coalesced_team.size() + coalesced_team.thread_rank()-1) % coalesced_team.size());

         if (coalesced_team.thread_rank() == 0){
            prev_val = gallatin::utils::typed_atomic_exchange(internal_reference, prev_val);
         }

         return prev_val;

      }


      __device__ T atomicCAS(T expected, T new_obj){

         return gallatin::utils::typed_atomic_CAS(internal_reference, expected, new_obj);

      }


      __device__ T load_acq(){

         return gallatin::utils::ld_acq(internal_reference);

      }


      //make templatized?
      __device__ T apply_rmw(T (*RMW)(T)){


         while (true){

            auto coalesced_team = cg::coalesced_threads();

            T original_read_val = cg::invoke_one_broadcast(coalesced_team, [&]() { return load_acq();});

            T read_val = original_read_val;

            T my_read_val;

            for (int i = 0; i < coalesced_team.size(); i++){


               if (coalesced_team.thread_rank() == i){

                  my_read_val = read_val;
                  read_val = RMW(read_val);
               }


               read_val = coalesced_team.shfl(read_val, i);

            }

            __threadfence();

            //at this point read_val is the final value of the operation.

            bool success = false;
            
            if (coalesced_team.thread_rank() == 0){

               success = (atomicCAS(original_read_val, read_val) == original_read_val);

            } 

            success = coalesced_team.ballot(success);

            if (success) return my_read_val;

            __threadfence();

         }

      }


      //TODO? make templatized
      __device__ void apply_mutate_exchange(T * my_arg, bool (*mutate_fn)(T *, T *, bool)){


         

         auto coalesced_team = cg::coalesced_threads();


         //shuffle arguments to all threads
         T * previous_thread_input = coalesced_team.shfl(my_arg, (coalesced_team.size() + coalesced_team.thread_rank()-1) % coalesced_team.size());


         if (coalesced_team.thread_rank() != 0){

            if (!mutate_fn(previous_thread_input, my_arg, false)){
               //should never be reached - this would mean non-exclusive access for these threads.

               //printf("Should not occur\n");
               asm volatile("trap;");
            }

         }

         //chain is built - all threads but 0 set.

         coalesced_team.sync();
         __threadfence();

         while (true){

            bool ballot = false;

            if (coalesced_team.thread_rank() == 0){

               T * head_node = (T *) gallatin::utils::ld_acq((uint64_t *)&internal_reference);

               ballot = mutate_fn(head_node, my_arg, true);

               if (ballot){

                  gallatin::utils::typed_atomic_exchange((uint64_t *)&internal_reference, (uint64_t)previous_thread_input);


               }

            }

            if (coalesced_team.ballot(ballot)) return;

            __threadfence();

         }


      }


   };


}  // namespace gallatin

#endif  // GPU_BLOCK_
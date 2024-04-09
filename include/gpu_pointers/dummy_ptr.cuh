#ifndef DUMMY_PTR
#define DUMMY_PTR

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
   struct dummy_pointer {

      using my_type = dummy_pointer<T>;

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

         cudaFree(host_version->internal_reference);

         cudaFreeHost(host_version);

      }

      __device__ T atomicAdd(T new_obj){

         return gallatin::utils::typed_atomic_add(internal_reference, new_obj);


      }

      __device__ T atomicExch(T new_obj){

         return gallatin::utils::typed_atomic_exchange(internal_reference, new_obj);

      }


      __device__ T atomicCAS(T expected, T new_obj){

         return gallatin::utils::typed_atomic_CAS(internal_reference, expected, new_obj);

      }


      __device__ T load_acq(){

         return gallatin::utils::ld_acq(internal_reference);

      }

      __device__ T apply_rmw(T (* RMW)(T)){


         while (true){

            T read_val = load_acq();

            T next_val = RMW(read_val);

            if ((atomicCAS(read_val, next_val)) == read_val) return read_val;

         }



      }


   };


}  // namespace gallatin

#endif  // GPU_BLOCK_
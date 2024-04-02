#ifndef DUMMY_HT
#define DUMMY_HT

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <cooperative_groups/scan.h>
#include <cuda.h>
#include <cuda_runtime_api.h>

#include <gallatin/allocators/alloc_utils.cuh>

#include <cuda/atomic>
#include <cuda/std/atomic>

#include "assert.h"
#include "stdio.h"

namespace cg = cooperative_groups;

// helper_macro
// define macros
#define MAX_VALUE(nbits) ((1ULL << (nbits)) - 1)
#define BITMASK(nbits) ((nbits) == 64 ? 0xffffffffffffffff : MAX_VALUE(nbits))

#define SET_BIT_MASK(index) ((1ULL << index))

// a pointer list managing a set section of device memory



//cache protocol
//query cache
//on success add to pin?
//need delete from potential buckets implementation - need to download warpcore...
//buidld with primary p2bht first.



namespace hashing_project {

namespace wrappers {

   template <typename Key, typename Val>
   struct packed_pair {
      Key first;
      Val second;
   };


   template <typename Key, typename Val, uint bucket_size>
   struct dummy_ht {


      using tile_type = cg::thread_block_tile<bucket_size>;

      using my_type = dummy_ht<Key, Val, bucket_size>;

      using packed_pair_type = packed_pair<Key, Val>;

      //dummy handle
      static __host__ my_type * generate_on_device(uint64_t cache_capacity, Key sentinel_key, Key tombstone_key, Val sentinel_val){

         return gallatin::utils::get_device_version<my_type>();

      }

      static __host__ void free_on_device(my_type * device_version){
         cudaFree(device_version);
      }

      //nope! no storage
      __device__ bool find_with_reference(tile_type my_tile, Key key, Val & val){
         return false;
      }

      __device__ bool upsert(tile_type my_tile, Key old_key, Val old_val, Key new_key, Val new_val){
         return false;
      }

      __device__ bool upsert(tile_type my_tile, packed_pair_type old_pair, packed_pair_type new_pair){
         return false;
      }

      __device__ bool insert_if_not_exists(tile_type my_tile, Key key, Val val){
         return false;
      }
      
      __device__ packed_pair_type find_replaceable_pair(tile_type my_tile, Key key){
         packed_pair_type new_pair{Key{0}, Val{0}};

         return new_pair;
      }

      static __device__ packed_pair_type pack_together(Key key, Val val){
         return packed_pair_type{key, val};
      }

      static char * get_name(){
         return "empty table";
      }


   };



} //namespace wrappers

}  // namespace ht_project

#endif  // GPU_BLOCK_
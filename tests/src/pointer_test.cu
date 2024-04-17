/*
 * ============================================================================
 *
 *        Authors:  
 *                  Hunter McCoy <hjmccoy@lbl.gov
 *                  Yuvaraj Chesetti <chesetti@cs.utah.edu>
 *                  Ashish Tiwari <hi@aashishtiwari.com.np>
 *
 * ============================================================================
 */





#include <gallatin/allocators/global_allocator.cuh>

#include <gallatin/allocators/timer.cuh>

#include <gpu_pointers/dummy_ptr.cuh>
#include <gpu_pointers/coalesce_ptr.cuh>


#include <stdio.h>
#include <iostream>
#include <assert.h>
#include <chrono>
#include <openssl/rand.h>



// #include <hashing_project/table_wrappers/p2_wrapper.cuh>
// #include <hashing_project/table_wrappers/dummy_ht.cuh>
// #include <hashing_project/table_wrappers/iht_wrapper.cuh>

#include <cooperative_groups.h>

namespace cg = cooperative_groups;

using namespace gallatin::allocators;

#define CHECK_CORRECTNESS 0


#if GALLATIN_DEBUG_PRINTS
   #define TEST_BLOCK_SIZE 256
#else
   #define TEST_BLOCK_SIZE 256
#endif





struct ll_node {

   uint64_t data;
   ll_node * next;

   __device__ void init(uint64_t tid){
      data = tid;
      next = nullptr;
      __threadfence();
   }

};


__device__ bool append_to_linked_list(ll_node * current_tail, ll_node * my_node, bool is_atomic=true){


   //printf("Inside function\n");

   //ll_node * next = (ll_node *) gallatin::utils::ld_acq((uint64_t *) &current_tail->next);

   if (is_atomic){

         current_tail = (ll_node *) atomicCAS((unsigned long long int *)&current_tail->next, 0ULL, (uint64_t) my_node);
         if (current_tail == 0ULL) return true;

         return false;

   } else {
         current_tail->next = my_node;
         __threadfence();
         return true;
   }


}




template <typename T>
__host__ T * generate_data(uint64_t nitems){


   //malloc space

   T * vals;

   cudaMallocHost((void **)&vals, sizeof(T)*nitems);


   //          100,000,000
   uint64_t cap = 100000000ULL;

   for (uint64_t to_fill = 0; to_fill < nitems; to_fill+=0){

      uint64_t togen = (nitems - to_fill > cap) ? cap : nitems - to_fill;


      RAND_bytes((unsigned char *) (vals + to_fill), togen * sizeof(T));



      to_fill += togen;

      //printf("Generated %llu/%llu\n", to_fill, nitems);

   }

   printf("Generation done\n");
   return vals;
}

template <template<typename> typename pointer_type, typename T>
__global__ void test_add_kernel(pointer_type<T> * ptr, uint64_t * bitarray, uint64_t n_ops){


   uint64_t tid = gallatin::utils::get_tid();

   if (tid >= n_ops) return;

   T my_value = 1;

   T old = ptr->atomicAdd(my_value);


   #if CHECK_CORRECTNESS

   uint64_t high = old/64;
   uint64_t low = old % 64;


   if (atomicOr((unsigned long long int *)&bitarray[high], (unsigned long long int) SET_BIT_MASK(low)) & SET_BIT_MASK(low)){
      printf("Double add to index %llu\n", old);
   }


   #endif


}


template <template<typename> typename pointer_type, typename T>
__global__ void test_exch_kernel(pointer_type<T> * ptr, uint64_t * bitarray, uint64_t n_ops){


   uint64_t tid = gallatin::utils::get_tid();

   if (tid >= n_ops) return;

   T my_value = tid;

   T old = ptr->atomicExch(my_value);


   #if CHECK_CORRECTNESS

   uint64_t high = old/64;
   uint64_t low = old % 64;


   if (atomicOr((unsigned long long int *)&bitarray[high], (unsigned long long int) SET_BIT_MASK(low)) & SET_BIT_MASK(low)){
      printf("Double add to index %llu\n", old);
   }


   #endif

}


template <template<typename> typename pointer_type, typename T>
__global__ void test_rmw_kernel(pointer_type<T> * ptr, uint64_t * bitarray, uint64_t n_ops){


   uint64_t tid = gallatin::utils::get_tid();

   if (tid >= n_ops) return;


   
   auto add_lambda = [](T a) { uint64_t ret_val = a; return ret_val;};

   T old = ptr->apply_rmw(add_lambda);


   #if CHECK_CORRECTNESS

   uint64_t high = old/64;
   uint64_t low = old % 64;


   if (atomicOr((unsigned long long int *)&bitarray[high], (unsigned long long int) SET_BIT_MASK(low)) & SET_BIT_MASK(low)){
      printf("Double add to index %llu\n", old);
   }


   #endif

   //printf("Done with %lu\n", tid);

}


template <template<typename> typename pointer_type>
__global__ void test_mutate_kernel(pointer_type<ll_node> * ptr, uint64_t n_ops){

   uint64_t tid = gallatin::utils::get_tid();

   if (tid >= n_ops) return;

   ll_node * my_node = (ll_node *) gallatin::allocators::global_malloc(sizeof(ll_node));

   if (my_node == nullptr){
      printf("Nullptr\n");
   }


   my_node->init(tid);


   ptr->apply_mutate_exchange(my_node, append_to_linked_list);

}



template <template<typename> typename pointer_type>
__host__ void ll_test(uint64_t n_ops){


   using ptr_type = pointer_type<ll_node>;

   //ll_node * starter_node = gallatin::utils::get_host_version<ll_node>();

   //starter_node->next = nullptr;

   ptr_type * dev_ptr = ptr_type::generate_on_device(ll_node{0ULL,nullptr});

   //cudaFreeHost(starter_node);



   gallatin::utils::timer add_timer;

   test_mutate_kernel<pointer_type><<<(n_ops-1)/512+1,512>>>(dev_ptr, n_ops);

   add_timer.sync_end();

   add_timer.print_throughput("Appended", n_ops);


   ptr_type::free_on_device(dev_ptr);

   //cudaFree(access_data);

}



//pull from blocks
//this kernel tests correctness, and outputs misses in a counter.
//works on actual pointers instead of uint64_t
//The correctness check is done by treating each allocation as a uint64_t and writing the tid
// if TID is not what is expected, we know that a double malloc has occurred.
template <template<typename> typename pointer_type>
__host__ void ptr_add_test(uint64_t n_ops){


   using ptr_type = pointer_type<uint64_t>;

   ptr_type * dev_ptr = ptr_type::generate_on_device(0ULL);

   uint64_t n_lock_uints = (n_ops)/64+1;

   uint64_t * bitarray = gallatin::utils::get_device_version<uint64_t>(n_lock_uints);

   cudaMemset(bitarray, 0ULL, sizeof(uint64_t)*n_lock_uints);




   gallatin::utils::timer add_timer;

   test_add_kernel<pointer_type, uint64_t><<<(n_ops-1)/1024+1,1024>>>(dev_ptr, bitarray, n_ops);

   add_timer.sync_end();

   add_timer.print_throughput("Added", n_ops);


   ptr_type::free_on_device(dev_ptr);
   cudaFree(bitarray);

   //cudaFree(access_data);

}


template <template<typename> typename pointer_type>
__host__ void ptr_exch_test(uint64_t n_ops){


   using ptr_type = pointer_type<uint64_t>;

   ptr_type * dev_ptr = ptr_type::generate_on_device(n_ops);

   uint64_t n_lock_uints = (n_ops)/64+1;

   uint64_t * bitarray = gallatin::utils::get_device_version<uint64_t>(n_lock_uints);

   cudaMemset(bitarray, 0ULL, sizeof(uint64_t)*n_lock_uints);




   gallatin::utils::timer add_timer;

   test_exch_kernel<pointer_type, uint64_t><<<(n_ops-1)/1024+1,1024>>>(dev_ptr, bitarray, n_ops);

   add_timer.sync_end();

   add_timer.print_throughput("Exchanged", n_ops);


   ptr_type::free_on_device(dev_ptr);
   cudaFree(bitarray);

   //cudaFree(access_data);

}

template <template<typename> typename pointer_type>
__host__ void ptr_rmw_test(uint64_t n_ops){


   using ptr_type = pointer_type<uint64_t>;

   ptr_type * dev_ptr = ptr_type::generate_on_device(0ULL);

   uint64_t n_lock_uints = (n_ops)/64+1;

   uint64_t * bitarray = gallatin::utils::get_device_version<uint64_t>(n_lock_uints);

   cudaMemset(bitarray, 0ULL, sizeof(uint64_t)*n_lock_uints);




   gallatin::utils::timer add_timer;

   test_rmw_kernel<pointer_type, uint64_t><<<(n_ops-1)/512+1,512>>>(dev_ptr, bitarray, n_ops);

   add_timer.sync_end();

   add_timer.print_throughput("RMW'ed", n_ops);


   ptr_type::free_on_device(dev_ptr);
   cudaFree(bitarray);

   //cudaFree(access_data);

}


int main(int argc, char** argv) {

   uint64_t n_ops;



   if (argc < 2){
      n_ops = 1000000;
   } else {
      n_ops = std::stoull(argv[1]);
   }


   printf("Dummy ptr\n");
   // ptr_add_test<gpu_pointers::dummy_pointer>(n_ops);

   // ptr_exch_test<gpu_pointers::dummy_pointer>(n_ops);

   // ptr_rmw_test<gpu_pointers::dummy_pointer>(1000000);


   // printf("Coalesced ptr\n");
   // ptr_add_test<gpu_pointers::coalesce_pointer>(n_ops);

   // ptr_exch_test<gpu_pointers::coalesce_pointer>(n_ops);

   // ptr_rmw_test<gpu_pointers::dummy_pointer>(n_ops);
   // ptr_rmw_test<gpu_pointers::coalesce_pointer>(n_ops);


   gallatin::allocators::init_global_allocator(24ULL*1024*1024*1024, 42);

   ll_test<gpu_pointers::dummy_pointer>(n_ops);
   ll_test<gpu_pointers::coalesce_pointer>(n_ops);

   gallatin::allocators::free_global_allocator();

   cudaDeviceReset();
   return 0;

}

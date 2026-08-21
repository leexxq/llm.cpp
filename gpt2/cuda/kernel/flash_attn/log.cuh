#pragma once
//for forward
#define fwd_debug_thread 0
// #define debug_forward

#ifdef debug_forward
	#define fwd_threadid_print(id,...) do{\
			if(thread(id)){\
				printf("[line: %d, thread:%d]",__LINE__,id);\
				printf(__VA_ARGS__);\
			}\
		}while(false)

	#define fwd_threadid_print_tensor(id,desc,tensor) do{\
			fwd_threadid_print(id,desc);\
			if(thread(id)){\
				print_tensor(tensor);\
			}\
		}while(false)

	#define fwd_thread_print(...) fwd_threadid_print(fwd_debug_thread,__VA_ARGS__)

	#define fwd_thread_print_tensor(desc,tensor) fwd_threadid_print_tensor(fwd_debug_thread,desc,tensor)
	#define fwd_thread_print_tensor_verbose(tensor) fwd_thread_print_tensor(#tensor" :",tensor)
#else
	#define fwd_thread_print(...)
	#define fwd_thread_print_tensor(x,y)
	#define fwd_thread_print_tensor_verbose(tensor)
#endif

//for backward

#define bwd_debug_thread 0
// #define bwd_debug_backward

#ifdef bwd_debug_backward
	#define bwd_threadid_print(id,...) do{\
			if(thread(id)){\
				printf("[line: %d, thread:%d]",__LINE__,id);\
				printf(__VA_ARGS__);\
			}\
		}while(false)

	#define bwd_threadid_print_tensor(id,desc,tensor) do{\
			bwd_threadid_print(id,desc);\
			if(thread(id)){\
				print_tensor(tensor);\
			}\
		}while(false)

	#define bwd_thread_print(...) bwd_threadid_print(bwd_debug_thread,__VA_ARGS__)

	#define bwd_thread_print_tensor(desc,tensor) bwd_threadid_print_tensor(bwd_debug_thread,desc,tensor)
	#define bwd_thread_print_tensor_verbose(tensor) bwd_thread_print_tensor(#tensor" :",tensor)
#else
	#define bwd_thread_print(...)
	#define bwd_thread_print_tensor(x,y)
	#define bwd_thread_print_tensor_verbose(tensor)
#endif

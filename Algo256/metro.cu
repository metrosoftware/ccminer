/*
 * Keccak 256
 *
 */

 extern "C"
 {
 #include "sph/sph_shavite.h"
 #include "sph/sph_simd.h"
 #include "sph/sph_keccak.h"
 }
 #include "miner.h"
 
 
 #include "cuda_helper.h"
 
 extern void keccak256_cpu_init(int thr_id, uint32_t threads);
 extern void keccak256_setBlock_M(int thr_id, void *pdata,const void *ptarget);
 extern void keccak256_cpu_hash_M(int thr_id, uint32_t threads, uint32_t startNounce, uint32_t *h_nounce);
 
 // CPU Hash
 void keccak256_general_hash(void *state, const void *input)
 {
	 sph_keccak_context ctx_keccak;
 
	 uint32_t hash[16];
 
	 sph_keccak256_init(&ctx_keccak);
	 sph_keccak256 (&ctx_keccak, input, len);
	 sph_keccak256_close(&ctx_keccak, (void*) hash);
 
	 memcpy(state, hash, 32);
 }

 void keccak256_metro_hash(void *state, const void *input)
 {
	 keccak256_general_hash(state, input, 98);
 }

 extern void setnounce(uint32_t *pdata, uint32_t nounce) {
    pdata[23] = (pdata[23] & 0xFFFFU) | ((nounce & 0xFFFFU) << 16);
    pdata[24] = (nounce & 0xFFFF0000U) >> 16;
}

extern uint32_t getnounce(uint32_t *pdata) {
    return ((pdata[23] >> 16) & 0xFFFFU) + ((pdata[24] & 0xFFFFU) << 16);
}

 extern int scanhash_keccak256_metro(int thr_id, uint32_t *pdata,
	 uint32_t *ptarget, uint32_t max_nonce,
	 uint32_t *hashes_done)
 {
	 static THREAD uint32_t *h_nounce = nullptr;
 
	 const uint32_t first_nonce = getnounce(pdata);
	 uint32_t intensity = (device_sm[device_map[thr_id]] > 500) ? 1 << 28 : 1 << 27;;
	 uint32_t throughputmax = device_intensity(device_map[thr_id], __func__, intensity); // 256*4096
	 uint32_t throughput = min(throughputmax, max_nonce - first_nonce) & 0xfffffc00;
 
 
	 if (opt_benchmark)
		 ptarget[7] = 0x0002;
 
	 static THREAD volatile bool init = false;
	 if(!init)
	 {
		 if(throughputmax == intensity)
			 applog(LOG_INFO, "GPU #%d: using default intensity %.3f", device_map[thr_id], throughput2intensity(throughputmax));
		 CUDA_SAFE_CALL(cudaSetDevice(device_map[thr_id]));
		 CUDA_SAFE_CALL(cudaDeviceReset());
		 CUDA_SAFE_CALL(cudaSetDeviceFlags(cudaschedule));
		 CUDA_SAFE_CALL(cudaDeviceSetCacheConfig(cudaFuncCachePreferL1));
		 CUDA_SAFE_CALL(cudaStreamCreate(&gpustream[thr_id]));
		 CUDA_SAFE_CALL(cudaMallocHost(&h_nounce, 2 * sizeof(uint32_t)));
		 keccak256_cpu_init(thr_id, (int)throughputmax);
 //		CUDA_SAFE_CALL(cudaMallocHost(&h_nounce, 2 * sizeof(uint32_t)));
		 mining_has_stopped[thr_id] = false;
		 init = true;
	 }
 
	 uint32_t endiandata[25];
     memcpy(&endiandata, pdata, 98);
 	 memset(((unsigned char*)&endiandata) + 98, 0, 2);

	 keccak256_setBlock_M(thr_id, (void*)endiandata, ptarget);
 
	 do {
 
		 keccak256_cpu_hash_M(thr_id, (int) throughput, getnounce(pdata), h_nounce);
		 if(stop_mining) {mining_has_stopped[thr_id] = true; cudaStreamDestroy(gpustream[thr_id]); pthread_exit(nullptr);}
		 if(h_nounce[0] != UINT32_MAX)
		 {
			 uint32_t Htarg = ptarget[7];
			 uint32_t vhash64[8]={0};
			 if(opt_verify){
	             *((uint32_t*)((unsigned char*)&endiandata[23] + 2)) = h_nounce[0];
                 keccak256_metro_hash(vhash64, endiandata);
			 }
			 if (vhash64[7] <= Htarg && fulltest(vhash64, ptarget))
			 {
				 int res = 1;
				 // check if there was some other ones...
				 *hashes_done = getnounce(pdata) - first_nonce + throughput;
				 if (h_nounce[1] != 0xffffffff)
				 {
					 if(opt_verify){
	                 *((uint32_t*)((unsigned char*)&endiandata[23] + 2)) = h_nounce[1];
					 keccak256_metro_hash(vhash64, endiandata);
 
					 }
					 if (vhash64[7] <= Htarg && fulltest(vhash64, ptarget))
					 {
						 pdata[26] = h_nounce[1];
						 res++;
						 if (opt_benchmark)
							 applog(LOG_INFO, "GPU #%d Found second nounce %08x", device_map[thr_id], h_nounce[1]);
					 }
					 else
					 {
						 if (vhash64[7] != Htarg)
						 {
							 applog(LOG_WARNING, "GPU #%d: result for %08x does not validate on CPU!", device_map[thr_id], h_nounce[1]);
						 }
					 }
				 }
				 setnounce(pdata, h_nounce[0]);
				 if (opt_benchmark)
					 applog(LOG_INFO, "GPU #%d Found nounce %08x", device_map[thr_id], h_nounce[0]);
				 return res;
			 }
			 else
			 {
				 if (vhash64[7] != Htarg)
				 {
					applog(LOG_WARNING, "GPU #%d: result for %08x does not validate on CPU!", device_map[thr_id], h_nounce[0]);
				}
			 }
		 }
 
		 setnounce(pdata, getnounce(pdata) + throughput); CUDA_SAFE_CALL(cudaGetLastError());
	 } while (!work_restart[thr_id].restart && ((uint64_t)max_nonce > ((uint64_t)(getnounce(pdata)) + (uint64_t)throughput)));
	 *hashes_done = getnounce(pdata) - first_nonce ;
	 return 0;
 }
 
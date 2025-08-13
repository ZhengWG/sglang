# default xccl envs for H20 from https://yuque.antfin.com/hegpb4/kg7h1z/rt5tbnz6bzzkmivw

if [ -z ${NCCL_SOCKET_IFNAME+x} ]; then export NCCL_SOCKET_IFNAME=bond0;fi
if [ -z ${NCCL_IB_HCA+x} ]; then export NCCL_IB_HCA=mlx5_bond;fi
if [ -z ${NVSHMEM_HCA_PE_MAPPING+x} ]; then export NVSHMEM_HCA_PE_MAPPING="mlx5_bond_0:1:2,mlx5_bond_1:1:2,mlx5_bond_2:1:2,mlx5_bond_3:1:2";fi
if [ -z ${GLOO_SOCKET_IFNAME+x} ]; then export GLOO_SOCKET_IFNAME=eth0;fi

if [ -z ${NCCL_NET_PLUGIN+x} ]; then export NCCL_NET_PLUGIN='';fi
if [ -z ${NCCL_IB_GID_INDEX+x} ]; then export NCCL_IB_GID_INDEX=3;fi
if [ -z ${NCCL_IB_TIMEOUT+x} ]; then export NCCL_IB_TIMEOUT=22;fi
if [ -z ${NCCL_IB_RETRY_CNT+x} ]; then export NCCL_IB_RETRY_CNT=7;fi
if [ -z ${NCCL_IB_SL+x} ]; then export NCCL_IB_SL=5;fi
if [ -z ${NCCL_IB_TC+x} ]; then export NCCL_IB_TC=136;fi
if [ -z ${NCCL_DEBUG+x} ]; then export NCCL_DEBUG=INFO;fi
if [ -z ${NCCL_SET_THREAD_NAME+x} ]; then export NCCL_SET_THREAD_NAME=1;fi
if [ -z ${NCCL_IB_QPS_PER_CONNECTION+x} ]; then export NCCL_IB_QPS_PER_CONNECTION=8;fi
if [ -z ${NCCL_SET_THREAD_NAME+x} ]; then export NCCL_SET_THREAD_NAME=1;fi
if [ -z ${NCCL_DEBUG_SUBSYS+x} ]; then export NCCL_DEBUG_SUBSYS=INIT,TUNING,GRAPH;fi

if [ -z ${NVSHMEM_DEBUG+x} ]; then export NVSHMEM_DEBUG=INFO;fi
if [ -z ${NVSHMEM_DEBUG_SUBSYS+x} ]; then export NVSHMEM_DEBUG_SUBSYS=INIT;fi
if [ -z ${NVSHMEM_IB_GID_INDEX+x} ]; then export NVSHMEM_IB_GID_INDEX=3;fi
if [ -z ${NVSHMEM_IB_TRAFFIC_CLASS+x} ]; then export NVSHMEM_IB_TRAFFIC_CLASS=16;fi
if [ -z ${NVSHMEM_IB_SL+x} ]; then export NVSHMEM_IB_SL=5;fi
if [ -z ${NVSHMEM_ENABLE_NIC_PE_MAPPING+x} ]; then export NVSHMEM_ENABLE_NIC_PE_MAPPING=1;fi

if [ -z ${IB_DEVICE_LIST+x} ]; then export IB_DEVICE_LIST="mlx5_bond_0,mlx5_bond_1,mlx5_bond_2,mlx5_bond_3";fi


"""Streaming attention with FP32 state across globally aligned K/V tiles.

The reverse 128-token traversal and unfused softmax scaling follow the
FlashAttention-2 numerical profile used by PyTorch on this BF16/64-dim model.
No per-shard BF16 attention output is produced before the final normalization.
This local Triton implementation does not import SGLang or FlashAttention.
"""
import torch
import triton
import triton.language as tl

@triton.jit
def _piece(Q,K,V,A,M,L,O, QS0:tl.constexpr,QS1:tl.constexpr,QS2:tl.constexpr,
           KS0:tl.constexpr,KS1:tl.constexpr,KS2:tl.constexpr,
           VS0:tl.constexpr,VS1:tl.constexpr,VS2:tl.constexpr,
           NQ:tl.constexpr,NK:tl.constexpr,H:tl.constexpr,D:tl.constexpr,SCALE:tl.constexpr,
           FIRST:tl.constexpr,FINAL:tl.constexpr,BM:tl.constexpr,BN:tl.constexpr,BD:tl.constexpr):
    rows=tl.program_id(0)*BM+tl.arange(0,BM)
    bh=tl.program_id(1); b=bh//H; h=bh%H
    ds=tl.arange(0,BD); ns=tl.arange(0,BN)
    q=tl.load(Q+b*QS0+rows[:,None]*QS1+h*QS2+ds[None,:],(rows[:,None]<NQ)&(ds[None,:]<D),0)
    off=(bh*NQ+rows[:,None])*D+ds[None,:]
    stat=bh*NQ+rows
    if FIRST:
        acc=tl.full((BM,BD),0,tl.float32)
        m=tl.full((BM,),-float('inf'),tl.float32)
        z0=tl.full((BM,),0,tl.float32)
        z1=tl.full((BM,),0,tl.float32)
        z2=tl.full((BM,),0,tl.float32)
        z3=tl.full((BM,),0,tl.float32)
    else:
        acc=tl.load(A+off,(rows[:,None]<NQ)&(ds[None,:]<D),0)
        m=tl.load(M+stat,rows<NQ,other=-float('inf'))
        z0=tl.load(L+stat*4+0,rows<NQ,0)
        z1=tl.load(L+stat*4+1,rows<NQ,0)
        z2=tl.load(L+stat*4+2,rows<NQ,0)
        z3=tl.load(L+stat*4+3,rows<NQ,0)
    for block in range(tl.cdiv(NK,BN)-1,-1,-1):
        keys=block*BN+ns
        k=tl.load(K+b*KS0+keys[None,:]*KS1+h*KS2+ds[:,None],(keys[None,:]<NK)&(ds[:,None]<D),0)
        v=tl.load(V+b*VS0+keys[:,None]*VS1+h*VS2+ds[None,:],(keys[:,None]<NK)&(ds[None,:]<D),0)
        s=tl.dot(q,k)
        s=tl.where(keys[None,:]<NK,s,-float('inf'))
        new_m=tl.maximum(m,tl.max(s,1))
        alpha=tl.exp2((m-new_m)*SCALE)
        p=tl.exp2(s*SCALE-new_m[:,None]*SCALE)
        acc=acc*alpha[:,None]
        for j in tl.static_range(BN//4):
            p0=tl.sum(tl.where(ns[None,:]==(j//2)*8+j%2+0,p,0.),1)
            if j == 0:
                z0=tl.fma(z0,alpha,p0)
            else:
                z0=z0+p0
            p1=tl.sum(tl.where(ns[None,:]==(j//2)*8+j%2+2,p,0.),1)
            if j == 0:
                z1=tl.fma(z1,alpha,p1)
            else:
                z1=z1+p1
            p2=tl.sum(tl.where(ns[None,:]==(j//2)*8+j%2+4,p,0.),1)
            if j == 0:
                z2=tl.fma(z2,alpha,p2)
            else:
                z2=z2+p2
            p3=tl.sum(tl.where(ns[None,:]==(j//2)*8+j%2+6,p,0.),1)
            if j == 0:
                z3=tl.fma(z3,alpha,p3)
            else:
                z3=z3+p3
        acc=tl.dot(p.to(q.dtype),v,acc)
        m=new_m
    if FINAL:
        denom=(z0+z2)+(z1+z3)
        # PyTorch's FlashAttention uses a correctly rounded FP32 reciprocal.
        # Triton's default division is approximate and changes a few BF16
        # rounding decisions, which can grow across diffusion steps.
        inverse=tl.inline_asm_elementwise("rcp.rn.f32 $0, $1;", constraints="=f,f",
                                         args=[denom], dtype=tl.float32, is_pure=True, pack=1)
        output=acc*inverse[:,None]
        tl.store(O+off,output,(rows[:,None]<NQ)&(ds[None,:]<D))
    else:
        tl.store(A+off,acc,(rows[:,None]<NQ)&(ds[None,:]<D))
        tl.store(M+stat,m,rows<NQ)
        tl.store(L+stat*4+0,z0,rows<NQ)
        tl.store(L+stat*4+1,z1,rows<NQ)
        tl.store(L+stat*4+2,z2,rows<NQ)
        tl.store(L+stat*4+3,z3,rows<NQ)

def piece(q,k,v,state=None,final=False,scale=None):
    b,nq,h,d=q.shape
    first=state is None
    if first:
        a=torch.empty((b,h,nq,d),device=q.device,dtype=torch.float32)
        m=torch.empty((b,h,nq),device=q.device,dtype=torch.float32)
        l=torch.empty((b,h,nq,4),device=q.device,dtype=torch.float32)
        state=(a,m,l)
    out=torch.empty((b,h,nq,d),device=q.device,dtype=q.dtype) if final else state[0]
    _piece[(triton.cdiv(nq,128),b*h)](q,k,v,*state,out,*q.stride()[:3],*k.stride()[:3],*v.stride()[:3],
        nq,k.shape[1],h,d,(d**-.5 if scale is None else scale)*1.4426950408889634,first,final,128,128,triton.next_power_of_2(d),
        num_warps=4,num_stages=1,enable_fp_fusion=False)
    return out.transpose(1,2) if final else state

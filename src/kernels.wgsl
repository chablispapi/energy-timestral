struct Params { m:u32, n:u32, k:u32, op:u32, rows:u32, slots:u32, heads:u32, axis:u32, eps:f32, theta:f32, rotary:u32, unused:u32 }
@group(0) @binding(0) var<storage,read> a:array<f32>;
@group(0) @binding(1) var<storage,read> b:array<f32>;
@group(0) @binding(2) var<storage,read> c:array<f32>;
@group(0) @binding(3) var<storage,read_write> z:array<f32>;
@group(0) @binding(4) var<uniform> p:Params;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id:vec3<u32>) {
 let i=id.x;
 if (i>=p.unused) { return; }
 if (p.op==0u) { // linear: weights are [out,in], bias is always present (possibly zeros)
   let row=i/p.n; let col=i%p.n;
   var sum=c[col];
   for(var j=0u;j<p.k;j=j+1u) { sum=sum+a[row*p.k+j]*b[col*p.k+j]; }
   z[i]=sum;
 } else if(p.op==1u) { // RMS norm
   let row=i/p.n; var sum=0.0;
   for(var j=0u;j<p.n;j=j+1u) { let v=a[row*p.n+j]; sum=sum+v*v; }
   z[i]=a[i]*inverseSqrt(sum/f32(p.n)+p.eps)*b[i%p.n];
 } else if(p.op==2u) { z[i]=a[i]+b[i];
 } else if(p.op==3u) { z[i]=a[i]/(1.0+exp(-a[i]));
 } else if(p.op==4u) { // fused SwiGLU gate * silu(up)
   let row=i/p.n; let col=i%p.n;
   let up=a[row*2u*p.n+p.n+col];
   z[i]=a[row*2u*p.n+col]*up/(1.0+exp(-up));
 } else if(p.op==5u) { z[i]=a[i]/(1.0+exp(-b[i]));
 } else if(p.op==6u) { // QKV rotary, positions in b
   let d=p.n/3u; let hd=d/p.heads;
   let row=i/p.n; let channel=i%p.n; let kind=channel/d;
   let dim=(channel%d)/p.heads;
   if(kind==2u) { z[i]=a[i]; return; }
   let half=hd/2u; let freqdim=dim%half;
   var angle=0.0;
   if(freqdim<p.rotary/2u) { angle=b[row]/pow(p.theta, f32(2u*freqdim)/f32(p.rotary)); }
   var other=i+half*p.heads; var sign=-1.0;
   if(dim>=half) { other=i-half*p.heads; sign=1.0; }
   z[i]=a[i]*cos(angle)+sign*a[other]*sin(angle);
 } else if(p.op==7u) { // scores: a=QKV, b=mask
   let keys=p.k; let key=i%keys; let j=i/keys;
   let head=j%p.heads; let query=(j/p.heads)%p.n; let batch=j/(p.heads*p.n);
   let d=p.rows; let hd=d/p.heads; // rows reused for model width
   var qr=batch; var qs=query; var kr=batch; var ks=key;
   if(p.axis==1u) { qr=query; qs=batch; kr=key; ks=batch; }
   if(b[(batch*p.n+query)*keys+key]<0.5) { z[i]=-1e30; return; }
   var sum=0.0;
   for(var t=0u;t<hd;t=t+1u) {
     sum=sum+a[(qr*p.slots+qs)*3u*d+t*p.heads+head]*a[(kr*p.slots+ks)*3u*d+d+t*p.heads+head];
   }
   z[i]=sum/sqrt(f32(hd));
 } else if(p.op==8u) { // stable softmax and value aggregation, output [R,S,D]
   let d=p.n; let hd=d/p.heads; let dim=(i%d)%hd; let head=(i%d)/hd;
   let slot=(i/d)%p.slots; let row=i/(d*p.slots);
   var batch=row; var query=slot;
   if(p.axis==1u) { batch=slot; query=row; }
   var maxval=-1e30;
   let base=((batch*p.rows+query)*p.heads+head)*p.k; // rows reused for query count
   for(var j=0u;j<p.k;j=j+1u) { maxval=max(maxval,b[base+j]); }
   var total=0.0; var sum=0.0;
   for(var j=0u;j<p.k;j=j+1u) {
     let w=exp(b[base+j]-maxval); total=total+w;
     var vr=row; var vs=j;
     if(p.axis==1u) { vr=j; vs=slot; }
     sum=sum+w*a[(vr*p.slots+vs)*3u*d+2u*d+dim*p.heads+head];
   }
   z[i]=sum/total;
 }
}

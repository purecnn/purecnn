cmd = `g++ -fPIC -shared -o libconv2d.so conv2d.cpp ../git/OpenBLAS/libopenblas.a -lpthread`
println("Running $cmd")
run(cmd)
using Merlin
include("conv2d.jl") 
srand(1234)
x = rand(Float32,8,7,3,2)
w = rand(Float32,3,3,3,4) 
padding = (3,2); stride = (2,1) 
y = conv2d_forward(x, w, padding, stride)
gy = y
gx = zeros(x)
gw = zeros(w)
conv2d_backward!(x, gx, w, gw, gy, padding, stride)

mx = Var(x);
mk = (size(w,1),size(w,2))
mw = (size(w,3),size(w,4))
mf = mf = Conv(typeof(x[1]), mk, mw, stride=stride, paddims=padding)
mf.w.data = w
my = mf(mx)
mx.grad = zeros(mx.data)
mf.w.grad = zeros(mf.w.data)
my.df(y)
println("ydiff:$(findmax(abs(my.data-y))[1])")
println("gxdiff:$(findmax(abs(mx.grad-gx))[1])")
println("gwdiff:$(findmax(abs(mf.w.grad-gw))[1])")

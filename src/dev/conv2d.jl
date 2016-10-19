function conv2d_forward{T}(x::Array{T}, w::Array{T}, padding, stride; bias = nothing)
    y = similar(x,
        div((size(x,1)+2*padding[1]-size(w,1)), stride[1]) + 1,
        div((size(x,2)+2*padding[2]-size(w,2)), stride[2]) + 1,
        size(w,4),
        size(x,4))
    work = similar(x, size(x,3) * size(w,1) * size(w,2) * size(y,1) * size(y,2))
    pbias = bias==nothing ? Ptr{Void}(0) : bias

    if T == Cfloat
        ccall((:conv2d_f32, "libconv2d.so"),
            Void,
            (Ptr{Cint}, Ptr{Cfloat}, Ptr{Cint}, Ptr{Cfloat}, Ptr{Cfloat}, Ptr{Cint},
                Ptr{Cint}, Ptr{Cfloat}, Ptr{Cfloat}),
            Cint[size(x)...], x, Cint[size(w)...], w, pbias, Cint[padding...],
                Cint[stride...], y, work)
    elseif T == Cdouble
        ccall((:conv2d_forward_f64, "libconv2d.so"),
            Void,
            (Ptr{Cint}, Ptr{Cdouble}, Ptr{Cint}, Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cint},
                Ptr{Cint}, Ptr{Cdouble}, Ptr{Cdouble}),
            Cint[size(x)...], x, Cint[size(w)...], w, pbias, Cint[padding...],
                Cint[stride...], y, work)
    end
    y
end

function conv2d_backward!{T}(x::Array{T}, gx, w::Array{T}, gw, gy::Array{T}, padding, stride;
    bias = nothing, gbias = nothing)

    work = similar(x, size(x,3) * size(w,1) * size(w,2) * size(y,1) * size(y,2))
    pgx = gx==nothing ? Ptr{Void}(0) : gx
    pgw = gw==nothing ? Ptr{Void}(0) : gw
    pbias = bias==nothing ? Ptr{Void}(0) : bias
    pgbias = gbias==nothing ? Ptr{Void}(0) : gbias

    if T == Cfloat
        ccall((:conv2d_grad, "libconv2d.so"),
            Void,
            (Ptr{Cint}, Ptr{Cfloat}, Ptr{Cfloat}, Ptr{Cint}, Ptr{Cfloat}, Ptr{Cfloat},
                Ptr{Cfloat}, Ptr{Cfloat}, Ptr{Cfloat}, Ptr{Cint}, Ptr{Cint}, Ptr{Cfloat}),
            Cint[size(x)...], x, pgx, Cint[size(w)...], w, pgw, pbias, pgbias, gy,
                Cint[padding...], Cint[stride...], work)
    elseif T == Cdouble
        ccall((:conv2d_backward_f64, "libconv2d.so"),
            Void,
            (Ptr{Cint}, Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cint}, Ptr{Cdouble}, Ptr{Cdouble},
                Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cint}, Ptr{Cint}, Ptr{Cdouble}),
            Cint[size(x)...], x, pgx, Cint[size(w)...], w, pgw, pbias, pgbias, gy,
                Cint[padding...], Cint[stride...], work)
    end
end

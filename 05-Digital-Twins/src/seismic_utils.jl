
function build_source_receiver_geometry(n, d, idx_wb; params)
    (; setup_type, nsrc, nrec) = params

    # Set up source and receiver geometries.
    if setup_type == :surface
        xrange = (d[1], (n[1] - 1) * d[1])
        y = 0.0f0
        z = 10.0f0
        xsrc = range(xrange[1]; stop=xrange[2], length=nsrc)
        ysrc = range(y; stop=y, length=nsrc)
        zsrc = range(z; stop=z, length=nsrc)
        src_positions = (xsrc, ysrc, zsrc)

        y = 0.0f0
        z = (idx_wb - 1) * d[2]
        xrec = range(xrange[1]; stop=xrange[2], length=nrec)
        yrec = range(y; stop=y, length=nrec)
        zrec = range(z; stop=z, length=nrec)
        rec_positions = (xrec, yrec, zrec)
    elseif setup_type == :left_right
        x = 0.0f0
        y = 0.0f0
        zrange = (d[2], (n[2] - 1) * d[2])
        xsrc = range(x; stop=x, length=nsrc)
        ysrc = range(y; stop=y, length=nsrc)
        zsrc = range(zrange[1]; stop=zrange[2], length=nsrc)
        src_positions = (xsrc, ysrc, zsrc)

        x = (n[1] - 1) * d[1]
        y = 0.0f0
        xrec = range(x; stop=x, length=nrec)
        yrec = range(y; stop=y, length=nrec)
        zrec = range(zrange[1]; stop=zrange[2], length=nrec)
        rec_positions = (xrec, yrec, zrec)
    else
        error("Unknown setup_type $(setup_type)")
    end
    return src_positions, rec_positions
end

import numpy as np
import sys
import BVHSmooth.code.bvh as bvh
import BVHSmooth.code.freqfilter as freqfilter
import BVHSmooth.code.angle as angle
import BVHSmooth.code.spacefilter as spacefilter
from BVHSmooth.code.helper import progressbar


def smooth(input,output,filter="butterworth",uo=60,order=2,border=100,sigma=0.2):
    INPUT = input
    OUTPUT = output
    FILTER = filter
    if FILTER == "butterworth":
        ORDER = order
        U0 = uo
        BORDER = border
    if FILTER == "gaussian":
        SIGMA = sigma
        BORDER = border
    if FILTER == "average":
        M = int(ARGS["-m"])

    bvh_file = bvh.read_file(INPUT)

    for j in progressbar(range(len(bvh_file["ROTATIONS"][0,:,0])), "APPLYING BUTTERWORTH FILTER: ", 40):
        for i in range(3):
            v = angle.floats_to_degrees(bvh_file["ROTATIONS"][:,j,i])
            p = angle.degrees_to_polars(v)
            if FILTER == "average": f_filtered = spacefilter.apply_average(p, M)
            else:
                f = freqfilter.fft(p,BORDER)
                if FILTER == "gaussian": fil = freqfilter.gaussian_filter(len(f), SIGMA)
                if FILTER == "butterworth": fil = freqfilter.butter_worth_filter(len(f), U0, ORDER)
                f_filtered = freqfilter.apply_filter(f,fil)
                f_filtered = freqfilter.ifft(f_filtered,BORDER)
            p = angle.complexes_to_polars(f_filtered)
            nv = angle.polars_to_degrees(p)
            bvh_file["ROTATIONS"][:,j,i] = nv

    bvh.write_file(OUTPUT, bvh_file)

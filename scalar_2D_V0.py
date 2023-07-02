def composer_call():
    from fn_graph import Composer
    composer_1 = (
        Composer()
        .update(
            # list of custom functions goes here
            ForwardBiCGSTABFFT.M,
        )
        # .update_parameters(input_length_side=input_length_x_side)
        # .cache()
    )
    return composer_1


# from line_profiler import LineProfiler
# profiler = LineProfiler()
# profiler.add_function(ForwardBiCGSTABFFT.WavefieldSctCircle)
# profiler.run('ForwardBiCGSTABFFT.WavefieldSctCircle')
# profiler.print_stats()

# # @jit(nopython=True, parallel=True)
# # DO NOT REPORT THIS... COMPILATION TIME IS INCLUDED IN THE EXECUTION TIME!
# start = time.perf_counter()
# ForwardBiCGSTABFFT.ITERBiCGSTABw(CHI, u_inc, FFTG, N1, N2, Errcri, itmax)
# end = time.perf_counter()
# print("Elapsed (with compilation) = {}s".format((end - start)))

# # NOW THE FUNCTION IS COMPILED, RE-TIME IT EXECUTING FROM CACHE
# start = time.perf_counter()
# ForwardBiCGSTABFFT.ITERBiCGSTABw(CHI, u_inc, FFTG, N1, N2, Errcri, itmax)
# end = time.perf_counter()
# print("Elapsed (after compilation) = {}s".format((end - start)))


# if __name__ == '__main__':
#     start = ForwardBiCGSTABFFT.perf_counter()
#     load_array()
#     duration = ForwardBiCGSTABFFT.perf_counter() - start
#     print('load_array', duration)

#     start = ForwardBiCGSTABFFT.perf_counter()
#     load_file()
#     duration = ForwardBiCGSTABFFT.perf_counter() - start
#     print('load_file', duration)

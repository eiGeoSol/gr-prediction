def mexh(freq, length=0.512, dt=0.001):
    import numpy as np
    
    a = 0.251 / freq  # Scale (0.251 - Central frequency of initial wavelet (a = 1))
    t = np.linspace(-length / 2, length / 2, int(length / dt) + 1)
    wavelet_ = 2 / (np.sqrt(3) * np.pi ** (1 / 4)) * (1 - (t / a) ** 2) * np.exp(-(t / a) ** 2 / 2)
    wavelet_ *= 1 / np.sqrt(a)  # Normalize Energy
    return t, wavelet_


def cwt_decompose(sig, wavelet_length_=1., wavelet_dt_=0.001, frequency_range=(5, 150.), frequency_step=1.):
    import numpy as np

    decomposition = np.empty(shape=(len(sig) - int(wavelet_length_ / wavelet_dt_),
                                    int((frequency_range[1] - frequency_range[0]) / frequency_step)))

    for column_idx, freq in enumerate(np.arange(frequency_range[0], frequency_range[1], frequency_step)):
        time, wavelet = mexh(freq, length=wavelet_length_, dt=wavelet_dt_)
        decomposition[:, column_idx] = np.convolve(sig, wavelet, 'valid')

    return decomposition


def create_samples(trace, wavelet_len=1., wavelet_dt=0.001, f_range=(2.5, 100.), f_step=5.):
    """

    This code return cwt decomposed sample
    
    trace : input saismic trace 
    wavelet_len : wavelet total length
    wavelet_dt : wavelet time discretization
    f_range : pseudo-frequency range for CWT decomposition
    f_step : pseudo-frequency step for wavelet generation 

    """

    # Create CWT for trace of one sample
    decomposition = cwt_decompose(
        trace, wavelet_len, wavelet_dt, f_range, f_step
    )
    return decomposition

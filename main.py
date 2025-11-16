import numpy as np
import pyaudio
import launchpad_py as lp
from threading import Thread
import time

FREQUENCY_BANDS = {
    0: (20, 60),
    # 1: (60, 120),
    1: (40, 150),
    # 2: (120, 250),
    2: (120, 280),
    # 3: (251, 500),
    3: (250, 550),
    # 4: (501, 2_000),
    4: (500, 2_500),
    # 5: (2_001, 4_000),
    5: (2_000, 5_000),
    # 6: (4_001, 6_000),
    6: (4_000, 7_000),
    # 7: (6_001, 10_000),
    7: (6_000, 12_000),
    # 8: (10_001, 20_000),
    8: (10_000, 20_000),
}
SPECTRUM_BANDS = np.zeros(len(FREQUENCY_BANDS), dtype=float)
FORMAT = pyaudio.paInt16
CHANNELS = 2
RATE = 44100
CHUNK = 1024
MAX_FREQUENCY = 20_000
NORMALIZATION_FACTOR = 100_000


def process_audio(audio_buffer):
    numpydata = np.frombuffer(audio_buffer, dtype=np.int16)
    fft_output = np.fft.rfft(numpydata)
    magnitudes = np.abs(fft_output)
    bucket_width = 25.0
    num_buckets = MAX_FREQUENCY // int(bucket_width)

    # Vectorized frequency computation for rfft bins
    frequencies = np.fft.rfftfreq(len(numpydata), d=1.0 / RATE)
    valid_mask = frequencies < MAX_FREQUENCY
    bucket_indices = (frequencies[valid_mask] // bucket_width).astype(int)
    valid_bucket_mask = bucket_indices < num_buckets
    if bucket_indices.size:
        frequency_buckets = np.bincount(
            bucket_indices[valid_bucket_mask],
            weights=magnitudes[valid_mask][valid_bucket_mask],
            minlength=num_buckets,
        )
    else:
        frequency_buckets = np.zeros(num_buckets)

    # Vectorized band aggregation using cumulative sums
    band_bounds = np.array(
        [FREQUENCY_BANDS[i] for i in range(len(FREQUENCY_BANDS))], dtype=float
    )
    lower_indices = (band_bounds[:, 0] // bucket_width).astype(int)
    upper_indices = (band_bounds[:, 1] // bucket_width).astype(int)
    cumulative = np.concatenate(([0.0], frequency_buckets.cumsum()))
    values = cumulative[upper_indices] - cumulative[lower_indices]
    SPECTRUM_BANDS[:] = values // NORMALIZATION_FACTOR


def update_launchpad(lp):
    width = 9
    height = 9
    colors = np.array(
        [
            [63, 0, 0],
            [63, 31, 0],
            [63, 63, 0],
            [0, 63, 0],
            [0, 63, 63],
            [0, 0, 63],
            [31, 0, 63],
            [63, 0, 63],
            [63, 63, 63],
        ],
        dtype=int,
    )
    led_indices = np.arange(height)[:, None]  # shape (9,1)
    band_indices = np.arange(width)[None, :]  # shape (1,9)
    button_numbers = 11 + band_indices + led_indices * 10  # broadcast to (9,9)
    prev_mask = np.zeros((height, width), dtype=bool)
    while True:
        time.sleep(0.05)
        band_values = SPECTRUM_BANDS.astype(int)
        num_leds = np.minimum(band_values // 10, height)  # length 9
        lit_mask = led_indices < num_leds[None, :]
        changed_on = lit_mask & ~prev_mask
        changed_off = ~lit_mask & prev_mask
        for li, bi in np.argwhere(changed_on):
            r, g, b = colors[bi]
            lp.LedCtrlRaw(int(button_numbers[li, bi]), int(r), int(g), int(b))
        for li, bi in np.argwhere(changed_off):
            lp.LedCtrlRaw(int(button_numbers[li, bi]), 0, 0, 0)
        prev_mask = lit_mask


def detect(p):
    info = p.get_host_api_info_by_index(0)
    num_devices = info.get("deviceCount")
    bh = None
    for i in range(num_devices):
        device_info = p.get_device_info_by_host_api_device_index(0, i)
        if device_info["name"].startswith("BlackHole"):
            bh = i
    return bh


def stream_audio(p):
    stream = p.open(
        format=FORMAT,
        channels=CHANNELS,
        rate=RATE,
        input=True,
        input_device_index=detect(p),
        frames_per_buffer=CHUNK,
    )

    while True:
        data = stream.read(CHUNK)
        process_audio(data)


def main():
    lpx = lp.LaunchpadLPX()
    lpx.Open(1, "lpx")
    lpx.LedAllOn(0)
    Thread(target=update_launchpad, args=(lpx,), daemon=True).start()
    p = pyaudio.PyAudio()

    while True:
        try:
            stream_audio(p)
        except KeyboardInterrupt:
            p.terminate()
            lpx.Reset()
            break
        except OSError:
            print("Audio Stream Error - Reinitializing...")


if __name__ == "__main__":
    main()

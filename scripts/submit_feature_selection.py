import os
import tempfile
from time import sleep

# fmt: off
JOBS = [
    # ('fs-r0001-01', 'mignot,owners,normal', '08:00:00', '5', '1', '-d data/narco_features/avg_kw21_long_ahc -r 01 -k 20 -n 1 -s 01'),
    # ('fs-r0001-02', 'mignot,owners,normal', '08:00:00', '5', '1', '-d data/narco_features/avg_kw21_long_ahc -r 01 -k 20 -n 1 -s 02'),
    # ('fs-r0001-03', 'mignot,owners,normal', '08:00:00', '5', '1', '-d data/narco_features/avg_kw21_long_ahc -r 01 -k 20 -n 1 -s 03'),
    # ('fs-r0001-04', 'mignot,owners,normal', '08:00:00', '5', '1', '-d data/narco_features/avg_kw21_long_ahc -r 01 -k 20 -n 1 -s 04'),
    # ('fs-r0001-05', 'mignot,owners,normal', '08:00:00', '5', '1', '-d data/narco_features/avg_kw21_long_ahc -r 01 -k 20 -n 1 -s 05'),
    # ('fs-r0001-06', 'mignot,owners,normal', '08:00:00', '5', '1', '-d data/narco_features/avg_kw21_long_ahc -r 01 -k 20 -n 1 -s 06'),
    # ('fs-r0001-07', 'mignot,owners,normal', '08:00:00', '5', '1', '-d data/narco_features/avg_kw21_long_ahc -r 01 -k 20 -n 1 -s 07'),
    # ('fs-r0001-08', 'mignot,owners,normal', '08:00:00', '5', '1', '-d data/narco_features/avg_kw21_long_ahc -r 01 -k 20 -n 1 -s 08'),
    # ('fs-r0001-09', 'mignot,owners,normal', '08:00:00', '5', '1', '-d data/narco_features/avg_kw21_long_ahc -r 01 -k 20 -n 1 -s 09'),
    # ('fs-r0001-10', 'mignot,owners,normal', '08:00:00', '5', '1', '-d data/narco_features/avg_kw21_long_ahc -r 01 -k 20 -n 1 -s 10'),
    # ('fs-r0003-01', 'mignot,owners,normal', '08:00:00', '5', '1', '-d data/narco_features/avg_kw21_long_ahc -r 03 -k 20 -n 1 -s 01'),
    # ('fs-r0003-02', 'mignot,owners,normal', '08:00:00', '5', '1', '-d data/narco_features/avg_kw21_long_ahc -r 03 -k 20 -n 1 -s 02'),
    # ('fs-r0003-03', 'mignot,owners,normal', '08:00:00', '5', '1', '-d data/narco_features/avg_kw21_long_ahc -r 03 -k 20 -n 1 -s 03'),
    # ('fs-r0003-04', 'mignot,owners,normal', '08:00:00', '5', '1', '-d data/narco_features/avg_kw21_long_ahc -r 03 -k 20 -n 1 -s 04'),
    # ('fs-r0003-05', 'mignot,owners,normal', '08:00:00', '5', '1', '-d data/narco_features/avg_kw21_long_ahc -r 03 -k 20 -n 1 -s 05'),
    # ('fs-r0003-06', 'mignot,owners,normal', '08:00:00', '5', '1', '-d data/narco_features/avg_kw21_long_ahc -r 03 -k 20 -n 1 -s 06'),
    # ('fs-r0003-07', 'mignot,owners,normal', '08:00:00', '5', '1', '-d data/narco_features/avg_kw21_long_ahc -r 03 -k 20 -n 1 -s 07'),
    # ('fs-r0003-08', 'mignot,owners,normal', '08:00:00', '5', '1', '-d data/narco_features/avg_kw21_long_ahc -r 03 -k 20 -n 1 -s 08'),
    # ('fs-r0003-09', 'mignot,owners,normal', '08:00:00', '5', '1', '-d data/narco_features/avg_kw21_long_ahc -r 03 -k 20 -n 1 -s 09'),
    # ('fs-r0003-10', 'mignot,owners,normal', '08:00:00', '5', '1', '-d data/narco_features/avg_kw21_long_ahc -r 03 -k 20 -n 1 -s 10'),
    # ('fs-r0005-01', 'mignot,owners,normal', '08:00:00', '5', '1', '-d data/narco_features/avg_kw21_long_ahc -r 05 -k 20 -n 1 -s 01'),
    # ('fs-r0005-02', 'mignot,owners,normal', '08:00:00', '5', '1', '-d data/narco_features/avg_kw21_long_ahc -r 05 -k 20 -n 1 -s 02'),
    # ('fs-r0005-03', 'mignot,owners,normal', '08:00:00', '5', '1', '-d data/narco_features/avg_kw21_long_ahc -r 05 -k 20 -n 1 -s 03'),
    # ('fs-r0005-04', 'mignot,owners,normal', '08:00:00', '5', '1', '-d data/narco_features/avg_kw21_long_ahc -r 05 -k 20 -n 1 -s 04'),
    # ('fs-r0005-05', 'mignot,owners,normal', '08:00:00', '5', '1', '-d data/narco_features/avg_kw21_long_ahc -r 05 -k 20 -n 1 -s 05'),
    # ('fs-r0005-06', 'mignot,owners,normal', '08:00:00', '5', '1', '-d data/narco_features/avg_kw21_long_ahc -r 05 -k 20 -n 1 -s 06'),
    # ('fs-r0005-07', 'mignot,owners,normal', '08:00:00', '5', '1', '-d data/narco_features/avg_kw21_long_ahc -r 05 -k 20 -n 1 -s 07'),
    # ('fs-r0005-08', 'mignot,owners,normal', '08:00:00', '5', '1', '-d data/narco_features/avg_kw21_long_ahc -r 05 -k 20 -n 1 -s 08'),
    # ('fs-r0005-09', 'mignot,owners,normal', '08:00:00', '5', '1', '-d data/narco_features/avg_kw21_long_ahc -r 05 -k 20 -n 1 -s 09'),
    # ('fs-r0005-10', 'mignot,owners,normal', '08:00:00', '5', '1', '-d data/narco_features/avg_kw21_long_ahc -r 05 -k 20 -n 1 -s 10'),
    # ('fs-r0010-01', 'mignot,owners,normal', '08:00:00', '5', '1', '-d data/narco_features/avg_kw21_long_ahc -r 10 -k 20 -n 1 -s 01'),
    # ('fs-r0010-02', 'mignot,owners,normal', '08:00:00', '5', '1', '-d data/narco_features/avg_kw21_long_ahc -r 10 -k 20 -n 1 -s 02'),
    # ('fs-r0010-03', 'mignot,owners,normal', '08:00:00', '5', '1', '-d data/narco_features/avg_kw21_long_ahc -r 10 -k 20 -n 1 -s 03'),
    # ('fs-r0010-04', 'mignot,owners,normal', '08:00:00', '5', '1', '-d data/narco_features/avg_kw21_long_ahc -r 10 -k 20 -n 1 -s 04'),
    # ('fs-r0010-05', 'mignot,owners,normal', '08:00:00', '5', '1', '-d data/narco_features/avg_kw21_long_ahc -r 10 -k 20 -n 1 -s 05'),
    # ('fs-r0010-06', 'mignot,owners,normal', '08:00:00', '5', '1', '-d data/narco_features/avg_kw21_long_ahc -r 10 -k 20 -n 1 -s 06'),
    # ('fs-r0010-07', 'mignot,owners,normal', '08:00:00', '5', '1', '-d data/narco_features/avg_kw21_long_ahc -r 10 -k 20 -n 1 -s 07'),
    # ('fs-r0010-08', 'mignot,owners,normal', '08:00:00', '5', '1', '-d data/narco_features/avg_kw21_long_ahc -r 10 -k 20 -n 1 -s 08'),
    # ('fs-r0010-09', 'mignot,owners,normal', '08:00:00', '5', '1', '-d data/narco_features/avg_kw21_long_ahc -r 10 -k 20 -n 1 -s 09'),
    # ('fs-r0010-10', 'mignot,owners,normal', '08:00:00', '5', '1', '-d data/narco_features/avg_kw21_long_ahc -r 10 -k 20 -n 1 -s 10'),
    # ('fs-r0015-01', 'mignot,owners,normal', '00:10:00', '5', '1', '-d data/narco_features/avg_kw21_long_ahc -r 15 -k 20 -n 1 -s 01'),
    # ('fs-r0015-02', 'mignot,owners,normal', '00:10:00', '5', '1', '-d data/narco_features/avg_kw21_long_ahc -r 15 -k 20 -n 1 -s 02'),
    # ('fs-r0015-03', 'mignot,owners,normal', '00:10:00', '5', '1', '-d data/narco_features/avg_kw21_long_ahc -r 15 -k 20 -n 1 -s 03'),
    # ('fs-r0015-04', 'mignot,owners,normal', '00:10:00', '5', '1', '-d data/narco_features/avg_kw21_long_ahc -r 15 -k 20 -n 1 -s 04'),
    # ('fs-r0015-05', 'mignot,owners,normal', '00:10:00', '5', '1', '-d data/narco_features/avg_kw21_long_ahc -r 15 -k 20 -n 1 -s 05'),
    # ('fs-r0015-06', 'mignot,owners,normal', '00:10:00', '5', '1', '-d data/narco_features/avg_kw21_long_ahc -r 15 -k 20 -n 1 -s 06'),
    # ('fs-r0015-07', 'mignot,owners,normal', '00:10:00', '5', '1', '-d data/narco_features/avg_kw21_long_ahc -r 15 -k 20 -n 1 -s 07'),
    # ('fs-r0015-08', 'mignot,owners,normal', '00:10:00', '5', '1', '-d data/narco_features/avg_kw21_long_ahc -r 15 -k 20 -n 1 -s 08'),
    # ('fs-r0015-09', 'mignot,owners,normal', '00:10:00', '5', '1', '-d data/narco_features/avg_kw21_long_ahc -r 15 -k 20 -n 1 -s 09'),
    # ('fs-r0015-10', 'mignot,owners,normal', '00:10:00', '5', '1', '-d data/narco_features/avg_kw21_long_ahc -r 15 -k 20 -n 1 -s 10'),
    # ('fs-r0030-01', 'mignot,owners,normal', '00:10:00', '5', '1', '-d data/narco_features/avg_kw21_long_ahc -r 30 -k 20 -n 1 -s 01'),
    # ('fs-r0030-02', 'mignot,owners,normal', '00:10:00', '5', '1', '-d data/narco_features/avg_kw21_long_ahc -r 30 -k 20 -n 1 -s 02'),
    # ('fs-r0030-03', 'mignot,owners,normal', '00:10:00', '5', '1', '-d data/narco_features/avg_kw21_long_ahc -r 30 -k 20 -n 1 -s 03'),
    # ('fs-r0030-04', 'mignot,owners,normal', '00:10:00', '5', '1', '-d data/narco_features/avg_kw21_long_ahc -r 30 -k 20 -n 1 -s 04'),
    # ('fs-r0030-05', 'mignot,owners,normal', '00:10:00', '5', '1', '-d data/narco_features/avg_kw21_long_ahc -r 30 -k 20 -n 1 -s 05'),
    # ('fs-r0030-06', 'mignot,owners,normal', '00:10:00', '5', '1', '-d data/narco_features/avg_kw21_long_ahc -r 30 -k 20 -n 1 -s 06'),
    # ('fs-r0030-07', 'mignot,owners,normal', '00:10:00', '5', '1', '-d data/narco_features/avg_kw21_long_ahc -r 30 -k 20 -n 1 -s 07'),
    # ('fs-r0030-08', 'mignot,owners,normal', '00:10:00', '5', '1', '-d data/narco_features/avg_kw21_long_ahc -r 30 -k 20 -n 1 -s 08'),
    # ('fs-r0030-09', 'mignot,owners,normal', '00:10:00', '5', '1', '-d data/narco_features/avg_kw21_long_ahc -r 30 -k 20 -n 1 -s 09'),
    # ('fs-r0030-10', 'mignot,owners,normal', '00:10:00', '5', '1', '-d data/narco_features/avg_kw21_long_ahc -r 30 -k 20 -n 1 -s 10'),
    # ('fs-r0060-01', 'mignot,owners,normal', '00:10:00', '5', '1', '-d data/narco_features/avg_kw21_long_ahc -r 60 -k 20 -n 1 -s 01'),
    # ('fs-r0060-02', 'mignot,owners,normal', '00:10:00', '5', '1', '-d data/narco_features/avg_kw21_long_ahc -r 60 -k 20 -n 1 -s 02'),
    # ('fs-r0060-03', 'mignot,owners,normal', '00:10:00', '5', '1', '-d data/narco_features/avg_kw21_long_ahc -r 60 -k 20 -n 1 -s 03'),
    # ('fs-r0060-04', 'mignot,owners,normal', '00:10:00', '5', '1', '-d data/narco_features/avg_kw21_long_ahc -r 60 -k 20 -n 1 -s 04'),
    # ('fs-r0060-05', 'mignot,owners,normal', '00:10:00', '5', '1', '-d data/narco_features/avg_kw21_long_ahc -r 60 -k 20 -n 1 -s 05'),
    # ('fs-r0060-06', 'mignot,owners,normal', '00:10:00', '5', '1', '-d data/narco_features/avg_kw21_long_ahc -r 60 -k 20 -n 1 -s 06'),
    # ('fs-r0060-07', 'mignot,owners,normal', '00:10:00', '5', '1', '-d data/narco_features/avg_kw21_long_ahc -r 60 -k 20 -n 1 -s 07'),
    # ('fs-r0060-08', 'mignot,owners,normal', '00:10:00', '5', '1', '-d data/narco_features/avg_kw21_long_ahc -r 60 -k 20 -n 1 -s 08'),
    # ('fs-r0060-09', 'mignot,owners,normal', '00:10:00', '5', '1', '-d data/narco_features/avg_kw21_long_ahc -r 60 -k 20 -n 1 -s 09'),
    # ('fs-r0060-10', 'mignot,owners,normal', '00:10:00', '5', '1', '-d data/narco_features/avg_kw21_long_ahc -r 60 -k 20 -n 1 -s 10'),
    # ('fs-r0150-01', 'mignot,owners,normal', '00:10:00', '5', '1', '-d data/narco_features/avg_kw21_long_ahc -r 150 -k 20 -n 1 -s 01'),
    # ('fs-r0150-02', 'mignot,owners,normal', '00:10:00', '5', '1', '-d data/narco_features/avg_kw21_long_ahc -r 150 -k 20 -n 1 -s 02'),
    # ('fs-r0150-03', 'mignot,owners,normal', '00:10:00', '5', '1', '-d data/narco_features/avg_kw21_long_ahc -r 150 -k 20 -n 1 -s 03'),
    # ('fs-r0150-04', 'mignot,owners,normal', '00:10:00', '5', '1', '-d data/narco_features/avg_kw21_long_ahc -r 150 -k 20 -n 1 -s 04'),
    # ('fs-r0150-05', 'mignot,owners,normal', '00:10:00', '5', '1', '-d data/narco_features/avg_kw21_long_ahc -r 150 -k 20 -n 1 -s 05'),
    # ('fs-r0150-06', 'mignot,owners,normal', '00:10:00', '5', '1', '-d data/narco_features/avg_kw21_long_ahc -r 150 -k 20 -n 1 -s 06'),
    # ('fs-r0150-07', 'mignot,owners,normal', '00:10:00', '5', '1', '-d data/narco_features/avg_kw21_long_ahc -r 150 -k 20 -n 1 -s 07'),
    # ('fs-r0150-08', 'mignot,owners,normal', '00:10:00', '5', '1', '-d data/narco_features/avg_kw21_long_ahc -r 150 -k 20 -n 1 -s 08'),
    # ('fs-r0150-09', 'mignot,owners,normal', '00:10:00', '5', '1', '-d data/narco_features/avg_kw21_long_ahc -r 150 -k 20 -n 1 -s 09'),
    # ('fs-r0150-10', 'mignot,owners,normal', '00:10:00', '5', '1', '-d data/narco_features/avg_kw21_long_ahc -r 150 -k 20 -n 1 -s 10'),
    # ('fs-r0300-01', 'mignot,owners,normal', '00:10:00', '5', '1', '-d data/narco_features/avg_kw21_long_ahc -r 300 -k 20 -n 1 -s 01'),
    # ('fs-r0300-02', 'mignot,owners,normal', '00:10:00', '5', '1', '-d data/narco_features/avg_kw21_long_ahc -r 300 -k 20 -n 1 -s 02'),
    # ('fs-r0300-03', 'mignot,owners,normal', '00:10:00', '5', '1', '-d data/narco_features/avg_kw21_long_ahc -r 300 -k 20 -n 1 -s 03'),
    # ('fs-r0300-04', 'mignot,owners,normal', '00:10:00', '5', '1', '-d data/narco_features/avg_kw21_long_ahc -r 300 -k 20 -n 1 -s 04'),
    # ('fs-r0300-05', 'mignot,owners,normal', '00:10:00', '5', '1', '-d data/narco_features/avg_kw21_long_ahc -r 300 -k 20 -n 1 -s 05'),
    # ('fs-r0300-06', 'mignot,owners,normal', '00:10:00', '5', '1', '-d data/narco_features/avg_kw21_long_ahc -r 300 -k 20 -n 1 -s 06'),
    # ('fs-r0300-07', 'mignot,owners,normal', '00:10:00', '5', '1', '-d data/narco_features/avg_kw21_long_ahc -r 300 -k 20 -n 1 -s 07'),
    # ('fs-r0300-08', 'mignot,owners,normal', '00:10:00', '5', '1', '-d data/narco_features/avg_kw21_long_ahc -r 300 -k 20 -n 1 -s 08'),
    # ('fs-r0300-09', 'mignot,owners,normal', '00:10:00', '5', '1', '-d data/narco_features/avg_kw21_long_ahc -r 300 -k 20 -n 1 -s 09'),
    # ('fs-r0300-10', 'mignot,owners,normal', '00:10:00', '5', '1', '-d data/narco_features/avg_kw21_long_ahc -r 300 -k 20 -n 1 -s 10'),
    # ('fs-r0600-01', 'mignot,owners,normal', '00:10:00', '5', '1', '-d data/narco_features/avg_kw21_long_ahc -r 600 -k 20 -n 1 -s 01'),
    # ('fs-r0600-02', 'mignot,owners,normal', '00:10:00', '5', '1', '-d data/narco_features/avg_kw21_long_ahc -r 600 -k 20 -n 1 -s 02'),
    # ('fs-r0600-03', 'mignot,owners,normal', '00:10:00', '5', '1', '-d data/narco_features/avg_kw21_long_ahc -r 600 -k 20 -n 1 -s 03'),
    # ('fs-r0600-04', 'mignot,owners,normal', '00:10:00', '5', '1', '-d data/narco_features/avg_kw21_long_ahc -r 600 -k 20 -n 1 -s 04'),
    # ('fs-r0600-05', 'mignot,owners,normal', '00:10:00', '5', '1', '-d data/narco_features/avg_kw21_long_ahc -r 600 -k 20 -n 1 -s 05'),
    # ('fs-r0600-06', 'mignot,owners,normal', '00:10:00', '5', '1', '-d data/narco_features/avg_kw21_long_ahc -r 600 -k 20 -n 1 -s 06'),
    # ('fs-r0600-07', 'mignot,owners,normal', '00:10:00', '5', '1', '-d data/narco_features/avg_kw21_long_ahc -r 600 -k 20 -n 1 -s 07'),
    # ('fs-r0600-08', 'mignot,owners,normal', '00:10:00', '5', '1', '-d data/narco_features/avg_kw21_long_ahc -r 600 -k 20 -n 1 -s 08'),
    # ('fs-r0600-09', 'mignot,owners,normal', '00:10:00', '5', '1', '-d data/narco_features/avg_kw21_long_ahc -r 600 -k 20 -n 1 -s 09'),
    # ('fs-r0600-10', 'mignot,owners,normal', '00:10:00', '5', '1', '-d data/narco_features/avg_kw21_long_ahc -r 600 -k 20 -n 1 -s 10'),
    # ('fs-r0900-01', 'mignot,owners,normal', '00:10:00', '5', '1', '-d data/narco_features/avg_kw21_long_ahc -r 900 -k 20 -n 1 -s 01'),
    # ('fs-r0900-02', 'mignot,owners,normal', '00:10:00', '5', '1', '-d data/narco_features/avg_kw21_long_ahc -r 900 -k 20 -n 1 -s 02'),
    # ('fs-r0900-03', 'mignot,owners,normal', '00:10:00', '5', '1', '-d data/narco_features/avg_kw21_long_ahc -r 900 -k 20 -n 1 -s 03'),
    # ('fs-r0900-04', 'mignot,owners,normal', '00:10:00', '5', '1', '-d data/narco_features/avg_kw21_long_ahc -r 900 -k 20 -n 1 -s 04'),
    # ('fs-r0900-05', 'mignot,owners,normal', '00:10:00', '5', '1', '-d data/narco_features/avg_kw21_long_ahc -r 900 -k 20 -n 1 -s 05'),
    # ('fs-r0900-06', 'mignot,owners,normal', '00:10:00', '5', '1', '-d data/narco_features/avg_kw21_long_ahc -r 900 -k 20 -n 1 -s 06'),
    # ('fs-r0900-07', 'mignot,owners,normal', '00:10:00', '5', '1', '-d data/narco_features/avg_kw21_long_ahc -r 900 -k 20 -n 1 -s 07'),
    # ('fs-r0900-08', 'mignot,owners,normal', '00:10:00', '5', '1', '-d data/narco_features/avg_kw21_long_ahc -r 900 -k 20 -n 1 -s 08'),
    # ('fs-r0900-09', 'mignot,owners,normal', '00:10:00', '5', '1', '-d data/narco_features/avg_kw21_long_ahc -r 900 -k 20 -n 1 -s 09'),
    # ('fs-r0900-10', 'mignot,owners,normal', '00:10:00', '5', '1', '-d data/narco_features/avg_kw21_long_ahc -r 900 -k 20 -n 1 -s 10'),
    # ('fs-r1800-01', 'mignot,owners,normal', '01:00:00', '5', '1', '-d data/narco_features/avg_kw21_long_ahc -r 1800 -k 20 -n 1 -s 01'),
    # ('fs-r1800-02', 'mignot,owners,normal', '01:00:00', '5', '1', '-d data/narco_features/avg_kw21_long_ahc -r 1800 -k 20 -n 1 -s 02'),
    # ('fs-r1800-03', 'mignot,owners,normal', '01:00:00', '5', '1', '-d data/narco_features/avg_kw21_long_ahc -r 1800 -k 20 -n 1 -s 03'),
    # ('fs-r1800-04', 'mignot,owners,normal', '01:00:00', '5', '1', '-d data/narco_features/avg_kw21_long_ahc -r 1800 -k 20 -n 1 -s 04'),
    # ('fs-r1800-05', 'mignot,owners,normal', '01:00:00', '5', '1', '-d data/narco_features/avg_kw21_long_ahc -r 1800 -k 20 -n 1 -s 05'),
    # ('fs-r1800-06', 'mignot,owners,normal', '01:00:00', '5', '1', '-d data/narco_features/avg_kw21_long_ahc -r 1800 -k 20 -n 1 -s 06'),
    # ('fs-r1800-07', 'mignot,owners,normal', '01:00:00', '5', '1', '-d data/narco_features/avg_kw21_long_ahc -r 1800 -k 20 -n 1 -s 07'),
    # ('fs-r1800-08', 'mignot,owners,normal', '01:00:00', '5', '1', '-d data/narco_features/avg_kw21_long_ahc -r 1800 -k 20 -n 1 -s 08'),
    # ('fs-r1800-09', 'mignot,owners,normal', '01:00:00', '5', '1', '-d data/narco_features/avg_kw21_long_ahc -r 1800 -k 20 -n 1 -s 09'),
    # ('fs-r1800-10', 'mignot,owners,normal', '01:00:00', '5', '1', '-d data/narco_features/avg_kw21_long_ahc -r 1800 -k 20 -n 1 -s 10'),
    # ('fs-r2700-01', 'mignot,owners,normal', '01:00:00', '5', '1', '-d data/narco_features/avg_kw21_long_ahc -r 2700 -k 20 -n 1 -s 01'),
    # ('fs-r2700-02', 'mignot,owners,normal', '01:00:00', '5', '1', '-d data/narco_features/avg_kw21_long_ahc -r 2700 -k 20 -n 1 -s 02'),
    # ('fs-r2700-03', 'mignot,owners,normal', '01:00:00', '5', '1', '-d data/narco_features/avg_kw21_long_ahc -r 2700 -k 20 -n 1 -s 03'),
    # ('fs-r2700-04', 'mignot,owners,normal', '01:00:00', '5', '1', '-d data/narco_features/avg_kw21_long_ahc -r 2700 -k 20 -n 1 -s 04'),
    # ('fs-r2700-05', 'mignot,owners,normal', '01:00:00', '5', '1', '-d data/narco_features/avg_kw21_long_ahc -r 2700 -k 20 -n 1 -s 05'),
    # ('fs-r2700-06', 'mignot,owners,normal', '01:00:00', '5', '1', '-d data/narco_features/avg_kw21_long_ahc -r 2700 -k 20 -n 1 -s 06'),
    # ('fs-r2700-07', 'mignot,owners,normal', '01:00:00', '5', '1', '-d data/narco_features/avg_kw21_long_ahc -r 2700 -k 20 -n 1 -s 07'),
    # ('fs-r2700-08', 'mignot,owners,normal', '01:00:00', '5', '1', '-d data/narco_features/avg_kw21_long_ahc -r 2700 -k 20 -n 1 -s 08'),
    # ('fs-r2700-09', 'mignot,owners,normal', '01:00:00', '5', '1', '-d data/narco_features/avg_kw21_long_ahc -r 2700 -k 20 -n 1 -s 09'),
    # ('fs-r2700-10', 'mignot,owners,normal', '01:00:00', '5', '1', '-d data/narco_features/avg_kw21_long_ahc -r 2700 -k 20 -n 1 -s 10'),
    # ('fs-r3600-01', 'mignot,owners,normal', '01:00:00', '5', '1', '-d data/narco_features/avg_kw21_long_ahc -r 3600 -k 20 -n 1 -s 01'),
    # ('fs-r3600-02', 'mignot,owners,normal', '01:00:00', '5', '1', '-d data/narco_features/avg_kw21_long_ahc -r 3600 -k 20 -n 1 -s 02'),
    # ('fs-r3600-03', 'mignot,owners,normal', '01:00:00', '5', '1', '-d data/narco_features/avg_kw21_long_ahc -r 3600 -k 20 -n 1 -s 03'),
    # ('fs-r3600-04', 'mignot,owners,normal', '01:00:00', '5', '1', '-d data/narco_features/avg_kw21_long_ahc -r 3600 -k 20 -n 1 -s 04'),
    # ('fs-r3600-05', 'mignot,owners,normal', '01:00:00', '5', '1', '-d data/narco_features/avg_kw21_long_ahc -r 3600 -k 20 -n 1 -s 05'),
    # ('fs-r3600-06', 'mignot,owners,normal', '01:00:00', '5', '1', '-d data/narco_features/avg_kw21_long_ahc -r 3600 -k 20 -n 1 -s 06'),
    # ('fs-r3600-07', 'mignot,owners,normal', '01:00:00', '5', '1', '-d data/narco_features/avg_kw21_long_ahc -r 3600 -k 20 -n 1 -s 07'),
    # ('fs-r3600-08', 'mignot,owners,normal', '01:00:00', '5', '1', '-d data/narco_features/avg_kw21_long_ahc -r 3600 -k 20 -n 1 -s 08'),
    # ('fs-r3600-09', 'mignot,owners,normal', '01:00:00', '5', '1', '-d data/narco_features/avg_kw21_long_ahc -r 3600 -k 20 -n 1 -s 09'),
    # ('fs-r3600-10', 'mignot,owners,normal', '01:00:00', '5', '1', '-d data/narco_features/avg_kw21_long_ahc -r 3600 -k 20 -n 1 -s 10'),
    # ('fs-r5400-01', 'mignot,owners,normal', '10:00:00', '5', '1', '-d data/narco_features/avg_kw21_long_ahc -r 5400 -k 20 -n 1 -s 01'),
    # ('fs-r5400-02', 'mignot,owners,normal', '10:00:00', '5', '1', '-d data/narco_features/avg_kw21_long_ahc -r 5400 -k 20 -n 1 -s 02'),
    # ('fs-r5400-03', 'mignot,owners,normal', '10:00:00', '5', '1', '-d data/narco_features/avg_kw21_long_ahc -r 5400 -k 20 -n 1 -s 03'),
    # ('fs-r5400-04', 'mignot,owners,normal', '10:00:00', '5', '1', '-d data/narco_features/avg_kw21_long_ahc -r 5400 -k 20 -n 1 -s 04'),
    # ('fs-r5400-05', 'mignot,owners,normal', '10:00:00', '5', '1', '-d data/narco_features/avg_kw21_long_ahc -r 5400 -k 20 -n 1 -s 05'),
    # ('fs-r5400-06', 'mignot,owners,normal', '10:00:00', '5', '1', '-d data/narco_features/avg_kw21_long_ahc -r 5400 -k 20 -n 1 -s 06'),
    # ('fs-r5400-07', 'mignot,owners,normal', '10:00:00', '5', '1', '-d data/narco_features/avg_kw21_long_ahc -r 5400 -k 20 -n 1 -s 07'),
    # ('fs-r5400-08', 'mignot,owners,normal', '10:00:00', '5', '1', '-d data/narco_features/avg_kw21_long_ahc -r 5400 -k 20 -n 1 -s 08'),
    # ('fs-r5400-09', 'mignot,owners,normal', '10:00:00', '5', '1', '-d data/narco_features/avg_kw21_long_ahc -r 5400 -k 20 -n 1 -s 09'),
    # ('fs-r5400-10', 'mignot,owners,normal', '10:00:00', '5', '1', '-d data/narco_features/avg_kw21_long_ahc -r 5400 -k 20 -n 1 -s 10'),
    ('fs-r7200-01', 'mignot,owners,normal', '1-00:00', '5', '1', '-d data/narco_features/avg_kw21_long_ahc -r 7200 -k 20 -n 1 -s 01'),
    ('fs-r7200-02', 'mignot,owners,normal', '1-00:00', '5', '1', '-d data/narco_features/avg_kw21_long_ahc -r 7200 -k 20 -n 1 -s 02'),
    ('fs-r7200-03', 'mignot,owners,normal', '1-00:00', '5', '1', '-d data/narco_features/avg_kw21_long_ahc -r 7200 -k 20 -n 1 -s 03'),
    ('fs-r7200-04', 'mignot,owners,normal', '1-00:00', '5', '1', '-d data/narco_features/avg_kw21_long_ahc -r 7200 -k 20 -n 1 -s 04'),
    ('fs-r7200-05', 'mignot,owners,normal', '1-00:00', '5', '1', '-d data/narco_features/avg_kw21_long_ahc -r 7200 -k 20 -n 1 -s 05'),
    ('fs-r7200-06', 'mignot,owners,normal', '1-00:00', '5', '1', '-d data/narco_features/avg_kw21_long_ahc -r 7200 -k 20 -n 1 -s 06'),
    ('fs-r7200-07', 'mignot,owners,normal', '1-00:00', '5', '1', '-d data/narco_features/avg_kw21_long_ahc -r 7200 -k 20 -n 1 -s 07'),
    ('fs-r7200-08', 'mignot,owners,normal', '1-00:00', '5', '1', '-d data/narco_features/avg_kw21_long_ahc -r 7200 -k 20 -n 1 -s 08'),
    ('fs-r7200-09', 'mignot,owners,normal', '1-00:00', '5', '1', '-d data/narco_features/avg_kw21_long_ahc -r 7200 -k 20 -n 1 -s 09'),
    ('fs-r7200-10', 'mignot,owners,normal', '1-00:00', '5', '1', '-d data/narco_features/avg_kw21_long_ahc -r 7200 -k 20 -n 1 -s 10'),
]
# fmt: on


def submit_job(jobname, partition, time, ncpus, gpus, option_str, *args):

    content = f"""#!/bin/bash
#
#SBATCH --job-name={jobname}
#SBATCH -p {partition}
#SBATCH --time={time}
#SBATCH --cpus-per-task={ncpus}
#SBATCH --output=/home/users/alexno/narcolepsy-detector/logs/{jobname}.out
#SBATCH --error=/home/users/alexno/narcolepsy-detector/logs/{jobname}.err
##################################################

cd $HOME/narcolepsy-detector
. activate_env.sh

python feature_selection.py {option_str}
"""

    #     print('')
    #     print('#######################################################################################')
    #     print(content)
    #     print('#######################################################################################')
    #     print('')
    with tempfile.NamedTemporaryFile(delete=False) as j:
        j.write(content.encode())
    os.system(f"sbatch {j.name}")


if __name__ == "__main__":

    print(f"Submitting {len(JOBS)} job(s) ...")

    for jobinfo in JOBS:
        submit_job(*jobinfo)

    print("All jobs have been submitted!")

# -*- coding: utf-8 -*-

from narco_biomarker import narco_biomarker
import biomarker_config
import glob
import matplotlib.pyplot as plt

p = '/home/jens/Documents/sleep_data'
d1 = glob.glob(p+"/*.edf")
d2 = glob.glob(p+"/*.EDF")

d = d1+d2

config = biomarker_config.Config()

biomarker = narco_biomarker(config)

biomarker.data_path = d[0]
print(biomarker.data_path)

H = biomarker.loadHeader()
print(H)

biomarker.channels_used['C3'] = 3
biomarker.channels_used['C4'] = []
biomarker.channels_used['O1'] = 4
biomarker.channels_used['O2'] = []
biomarker.channels_used['EOG-L'] = 0
biomarker.channels_used['EOG-R'] = 1
biomarker.channels_used['EMG'] = 5

biomarker.eval_all()

H = biomarker.get_hypnodensity()
"""
Copyright (c) Facebook, Inc. and its affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

import torchaudio as ta
import numpy as np


class BinauralDataset:

    '''
    dataset_directory: (str) base directory of the dataset
    chunk_size_ms: (int) length of an acoustic chunk in ms
    overlap: (float) overlap ratio between two neighboring audio chunks, must be in range [0, 1)
    '''
    def __init__(self,
                 dataset_directory,
                 chunk_size_ms=200,
                 overlap=0.5
                 ):
        super().__init__()
        # load audio data and relative transmitter/receiver position/orientation
        self.mono, self.binaural, self.view = [], [], []
        for subject_id in range(8):
            mono, _ = ta.load(f"{dataset_directory}/subject{subject_id + 1}/mono.wav")
            binaural, _ = ta.load(f"{dataset_directory}/subject{subject_id + 1}/binaural.wav")
            # receiver is fixed at origin in this dataset, so we only need transmitter view
            tx_view = np.loadtxt(f"{dataset_directory}/subject{subject_id + 1}/tx_positions.txt").transpose()
            self.mono.append(mono)
            self.binaural.append(binaural)
            self.view.append(tx_view.astype(np.float32))
        # ensure that chunk_size is a multiple of 400 to match audio (48kHz) and receiver/transmitter positions (120Hz)
        self.chunk_size = chunk_size_ms * 48
        if self.chunk_size % 400 > 0:
            self.chunk_size = self.chunk_size + 400 - self.chunk_size % 400
        # compute chunks
        self.chunks = []
        for subject_id in range(8):
            last_chunk_start_frame = self.mono[subject_id].shape[-1] - self.chunk_size + 1
            hop_length = int((1 - overlap) * self.chunk_size)
            for offset in range(0, last_chunk_start_frame, hop_length):
                self.chunks.append({'subject': subject_id, 'offset': offset})

    def __len__(self):
        '''
        :return: number of training chunks in dataset
        '''
        return len(self.chunks)

    def __getitem__(self, idx):
        '''
        :param idx: index of the chunk to be returned
        :return: mono audio as 1 x T tensor
                 binaural audio as 2 x T tensor
                 relative rx/tx position as 7 x K tensor, where K = T / 400 (120Hz tracking vs. 48000Hz audio)
        '''
        subject = self.chunks[idx]['subject']
        offset = self.chunks[idx]['offset']
        mono = self.mono[subject][:, offset:offset+self.chunk_size]
        binaural = self.binaural[subject][:, offset:offset+self.chunk_size]
        view = self.view[subject][:, offset//400:(offset+self.chunk_size)//400]
        return mono, binaural, view

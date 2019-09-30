import os

import cv2
import pandas as pd
from tqdm import tqdm
import numpy as np

from mlcomp.contrib.transform.rle import rle2mask
from mlcomp.worker.executors import Executor
from executors.preprocess import BASE_DIR

@Executor.register
class Masks(Executor):
    def work(self) -> dict:
        df = pd.read_csv('TEMP/fold.csv')
        os.makedirs(f'{BASE_DIR}/train_masks', exist_ok=True)

        df = df.sort_values(by='ImageId_ClassId')
        size = (256, 1600)
        mask = np.zeros(size)
        res = []
        for row in tqdm(df.itertuples(), total=df.shape[0]):
            pixels = row.EncodedPixels
            if not isinstance(pixels, str):
                pixels = ''

            mask_cls = rle2mask(pixels, size[::-1])
            mask[mask_cls > 0] = row.class_id

            if row.ImageId_ClassId.endswith('_4'):
                img_id = row.ImageId_ClassId.split('.')[0].strip()
                cv2.imwrite(f'{BASE_DIR}/train_masks/{img_id}.png', mask)
                mask = np.zeros(size)

                res.append(
                    {
                        'fold': row.fold,
                        'image': f'{img_id}.jpg',
                        'mask': f'{img_id}.png'
                    }
                )

        pd.DataFrame(res).to_csv(f'{BASE_DIR}/masks.csv', index=False)


if __name__ == '__main__':
    Masks().work()

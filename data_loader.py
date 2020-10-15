import numpy as np
import os
from PIL import Image
import pickle
import csv


class DataLoader:
    def __init__(self, path_to_data):
        self.path_to_data = path_to_data

    def resize_images(self, fname):
        return np.asarray(
            Image.open("data/Images/{}.jpg".format(fname)).resize((320, 448))
        )

    def pickle_dataset(self):
        images = []
        genres = []

        with open('data/train.csv') as data_csv:
            csv_reader = csv.reader(data_csv, delimiter=',')

            line = 0
            for row in csv_reader:
                if line > 0:
                    fname = row[0]

                    genres.append(row[2:].index('1'))
                    images.append(self.resize_images(fname))

                if (line+1) % 500 == 0:
                    print("Grabbed {} data points...".format(line+1))
                    break

                line += 1

        image_outfile = open("data/images.pkl", 'wb')
        genre_outfile = open("data/genres.pkl", 'wb')

        print("Dumping into pickle files...")
        pickle.dump(images, image_outfile)
        pickle.dump(genres, genre_outfile)

        image_outfile.close()
        genre_outfile.close()


if __name__ == "__main__":
    DataLoader = DataLoader("?")
    DataLoader.pickle_dataset()
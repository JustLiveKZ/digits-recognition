import argparse

import mnist
from train import train
from test import test


class Application(object):
    def __init__(self):
        self.args = None
        self.parser = argparse.ArgumentParser(prog='Handwritten digits recognitions')
        self.parser.add_argument('command', choices=['train', 'test'],
                                 help='command you want to perform, must be either "train" or "test"')

    def run(self):
        self.args = self.parser.parse_args()
        getattr(self, self.args.command)()

    def train(self):
        images, labels = mnist.read('training')
        train(images, labels)

    def test(self):
        images, labels = mnist.read('testing')
        test(images, labels)

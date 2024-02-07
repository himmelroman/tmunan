import enum

from fastapi import FastAPI

from tmunan.theatre.slideshow import Slideshow


class PerformanceType(enum.Enum):
    Slideshow = 0


def create_performance(name: PerformanceType, app: FastAPI):

    if name == PerformanceType.Slideshow:
        return Slideshow(app)

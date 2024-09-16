THIS IS A MULTI-YEAR EFFORT TO ISOLATE THE NORTHERN AURORA

INTRODUCTION
The aurora are a visual manifestation of how highly
energetic charged solar particles interact with the
Earth’s magnetic field. Beyond its beauty, the aurora
can influence the dynamics of the thermosphere. Its
effects in the thermosphere can be seen by the
changes in:

❖ Temperature
❖ Density and composition
❖ Wind patterns

By studying these effects, we gain insights of the
complexities and variability of the upper atmosphere.
GOLD data offer a unique advantage of taking
measurements within the FUV range, allowing for the
distinction of the auroras without contamination from
ground/city lights due to shorter wavelengths of light
being absorbed more easily in the atmosphere.

PURPOSE
With our increasing reliance on space infrastructure,
the necessity to understand the near-space
environment becomes increasingly important. Incoming
solar particles can:

❖ Damage spacecraft components
❖ Prompt premature de-orbiting of spacecraft
❖ Induce disturbances in the magnetosphere
    † Potentially leading to issues with power grid
systems on the ground

There are current scientific models require
measurements of incoming solar wind that can offer a
forecast of the aurora within a short window of time.
Data will be used to validate the accuracy of the
models.

METHODOLOGY
There are currently two different data processing approaches to isolating the northern aurora. Both processes utilize basic physical phenomenon to constrain multiple different conditions (vauge I know). The current implementation being developed relies on basic computer vision and unsupervised learning techniques. An example of a run through the algorithm shows complete detection of the aurora on the dayside of the globe (even we can't see it). However, the current implementation only seems to work for equinox. This is still progress towards the correct direction, as there was previously no implementation that was able to isolate the aurora at any point during the day.

The algorithm utilizes very basic k-means clustering to essentially group pixels that are at similar intensities with one another. A new clustering method, agglomerative clustering, is able to cluster pixels that are near intensity AND pixel location from one another. It essentially is taking into consideration the "connectvity" of a given cluster. SInce the aurora typically form some type of ovular shape this is extremely useful in being able to isolate the ring. 





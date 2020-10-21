#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
# Created By  : Angeliki Agathi Tsintzira
# Github      : https://github.com/AngelikiTsintzira
# Linkedin    : https://www.linkedin.com/in/angeliki-agathi-tsintzira/
# Created Date: October 2020
# =============================================================================
# Licence GPLv3
# =============================================================================
# This file is part of Continuous implicit authentication of smartphone users using navigation data.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
#
# =============================================================================
# Description
# =============================================================================
"""
This is a Python 3.7.4 64bit project.

This class contains features for each model/user.
It works as a database to save multiple objects that are characterized by the same things.

"""

class Features:
  
    # Statistic feautures based on time
    Μean = []
    STD = []
    Max = []
    Min = []
    Range = []
    Percentile25 = []
    Percentile50 = []
    Percentile75 = []
    Kurtosis = []
    Skewness = []
    Entropy = []

    # Statistics feautures based on frequency
    Amplitude1 = []
    Amplitude2 = []
    Frequency2 = []
    Mean_frequency = []

    # Gestures
    Dx = []
    Dy = []
    Vx = []
    Vy = []

    # Output
    y = []

    def __init__(self):
        # Statistic feautures based on time
        self.Μean = []
        self.STD = []
        self.Max = []
        self.Min = []
        self.Range = []
        self.Percentile25 = []
        self.Percentile50 = []
        self.Percentile75 = []
        self.Kurtosis = []
        self.Skewness = []
        self.Entropy = []
        self.y = []

        # Statistics feautures based on frequency
        self.Amplitude1 = []
        self.Amplitude2 = []
        self.Frequency2 = []
        self.Mean_frequency = []

        # Gestures
        self.Dx = []
        self.Dy = []
        self.Vx = []
        self.Vy = []

    # Set Methods
    def setΜean(self, value):
        self.Μean.append(value)
    
    def setSTD(self, value):
        self.STD.append(value)

    def setMax(self, value):
        self.Max.append(value)
    
    def setMin(self, value):
        self.Min.append(value)

    def setRange(self, value):
        self.Range.append(value)

    def setPercentile25(self, value):
        self.Percentile25.append(value)

    def setPercentile50(self, value):
        self.Percentile50.append(value)

    def setPercentile75(self, value):
        self.Percentile75.append(value)

    def setKurtosis(self, value):
        self.Kurtosis.append(value)

    def setSkewness(self, value):
        self.Skewness.append(value)

    def setEntropy(self, value):
        self.Entropy.append(value)

    def setAmplitude1(self, value):
        self.Amplitude1.append(value)

    def setAmplitude2(self, value):
        self.Amplitude2.append(value)

    def setFrequency2(self, value):
        self.Frequency2.append(value)

    def setMean_frequency(self, value):
        self.Mean_frequency.append(value)

    def setY(self, value):
        self.y.append(value)

    def setDx(self, value):
        self.Dx.append(value)  

    def setDy(self, value):
        self.Dy.append(value)

    def setVx(self, value):
        self.Vx.append(value)

    def setVy(self, value):
        self.Vy.append(value)

    # Get Methods
    def getMean(self):
        return self.Μean

    def getSTD(self):
        return self.STD
    
    def getMax(self):
        return self.Max
    
    def getMin(self):
        return self.Min

    def getRange(self):
        return self.Range

    def getPercentile25(self):
        return self.Percentile25

    def getPercentile50(self):
        return self.Percentile50

    def getPercentile75(self):
        return self.Percentile75

    def getKurtosis(self):
        return self.Kurtosis

    def getEntropy(self):
        return self.Entropy

    def getSkewness(self):
        return self.Skewness

    def getAmplitude1(self):
        return self.Amplitude1

    def getAmplitude2(self):
        return self.Amplitude2

    def getFrequency2(self):
        return self.Frequency2

    def getMean_frequency(self):
        return self.Mean_frequency

    def getY(self):
        return self.y

    def getDx(self):
        return self.Dx

    def getDy(self):
        return self.Dy

    def getVx(self):
        return self.Vx

    def getVy(self):
        return self.Vy
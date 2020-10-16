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

This class contains metrics for each model/user.
It works as a database to save multiple objects that are characterized by the same things.

"""
class Metrics:
  
    accuracy = []
    f1score = []
    FAR = []
    FRR = []
    ROC = []
    falseAccept = []
    falseReject= []
    trueAccept = []
    trueReject= []
    sizeTest = []

    def __init__(self):
        self.accuracy = []
        self.f1score = []
        self.FAR = []
        self.FRR = []
        self.ROC = []
        self.falseAccept = []
        self.falseReject= []
        self.trueAccept = []
        self.trueReject= []
        self.sizeTest = []

    # Set Methods
    def setAccuracy(self, value):
        self.accuracy.append(value)

    def setf1score(self, value):
        self.f1score.append(value)

    def setFAR(self, value):
        self.FAR.append(value)

    def setFRR(self, value):
        self.FRR.append(value)

    def setfalseAccept(self, value):
        self.falseAccept.append(value)
    
    def setfalseReject(self, value):
        self.falseReject.append(value)

    def settrueAccept(self, value):
        self.trueAccept.append(value)

    def settrueReject(self, value):
        self.trueReject.append(value)

    def setsizeTest(self, value):
        self.sizeTest.append(value)

    # Get Methods
    def getAccuracy(self):
        return self.accuracy

    def getf1score(self):
        return self.f1score

    def getFAR(self):
        return self.FAR

    def getFRR(self):
        return self.FRR

    def getfalseAccept(self):
        return self.falseAccept
    
    def getfalseReject(self):
        return self.falseReject

    def gettrueAccept(self):
        return self.trueAccept

    def gettrueReject(self):
        return self.trueReject

    def getsizeTest(self):
        return self.sizeTest
    
    
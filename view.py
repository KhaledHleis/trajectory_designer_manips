#!/usr/bin/env python3
"""
Quick entry point: python view.py [--dataset output/cable_survey_dataset]
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from viewer.viewer_app import main
main()

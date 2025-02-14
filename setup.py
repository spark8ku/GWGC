#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from setuptools import setup, find_packages

setup(
    name='functions_GWGC',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        "scikit-learn==1.4.1.post1", 
        "xgboost==2.0.3", 
        "catboost==1.2.7",
        "lightgbm==4.5.0",
        "bayesian-optimization"
    ],
    python_requires=">=3.8",
)


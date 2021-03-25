---
layout: post
title:  "Some Notes on dragonfly (bayesian optimization)"
date:   2021-03-20
categories: jekyll update
mathjax: true
---

# Domain

See: [config_parser.py](https://github.com/dragonfly/dragonfly/blob/master/dragonfly/parse/config_parser.py#137)

* "float": Needs ("min", "max"). {'type': 'float', 'min': -5.2, 'max': 5.7}

* "int": Needs ("min", "max"). {'type': 'int', 'min': -5, 'max': 3}

* "discrete": Needs [list of items]. {'type': 'discrete', 'items': ['a', 'b', 'c']}

* "discrete_numeric": Needs [list of items] or [range of items]. {'type': 'discrete_numeric', 'items': [1, 2, 3]} or {'type': 'discrete_numeric', 'items': '0.0:0.25:1'} where items are "startpoint:stepsize:endpoint". Note if passing explicit list, items need to be numeric.

* "boolean": {'type': 'boolean'}





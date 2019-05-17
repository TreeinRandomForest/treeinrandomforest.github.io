---
layout: post
title:  "Learning Optimization Algorithms to Tune Hardware Parameters - Part I"
date:   2019-05-16
categories: [machine-learning]
tags: [machine-learning, systems, reinforcement-learning]
mathjax: true
---

Modern computers are complex systems. They consist of an ensemble of parts, each specializing in its own domain, that have to work together to compute efficiently.

If one were to look closely at any one of these parts - operating systems, memory, processors, network cards etc., one would find hundreds or thousands of parameters that affect how that part operates and how it interacts with other sub-systems.

So it is natural to ask whether it is possible to tune these parameters for maximum performance. Maximum performance is a vague term and is eventually defined by the user. For example, it could mean lowest latency for a network card, or, energy efficiency or instructions per cycle for a processor.


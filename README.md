# BSP S5: Exploring the Forward-Forward Algorithm for Training Recurrent Neural Networks

This repository contains all important information about Bachelor's semester project in Semester 5, performed during Autumn 2024 by Fedor Chikhachev, under the supervision of Prof. Luis A. Leiva.

## Project report

A formal [report](BSP-S5-Forward-Forward-RNN.pdf) in IEEE format has been produced as a final deliverable for this project.

## Source code

Inside `experiments` folder, you can find 4 subfolders referring to 4 datasets which were used for the study: 1-dollar [[1]](#1), MCYT [[2]](#2), raton [[3]](#3), MobileTouchDB [[4]](#4). All these datasets were augmented with synthetic negative data produced with Sigma-Lognormal technique (cf. report). 
Files to run - `train_loop_ff_pos_neg.py` for Forward-Forward training algorithm, `train_loop_bp_pos_neg.py` for conventional Backpropagation through time (BPTT).

## References
<a id="1">[1]</a> 
Wobbrock JO, Wilson AD, Li Y. Gestures without libraries, toolkits or training: a $1 recognizer for user interface prototypes. InProceedings of the 20th annual ACM symposium on User interface software and technology 2007 Oct 7 (pp. 159-168).

<a id="2">[2]</a> 
Ortega-Garcia J, Fierrez-Aguilar J, Simon D, Gonzalez J, Faundez-Zanuy M, Espinosa V, Satue A, Hernaez I, Igarza JJ, Vivaracho C, Escudero D. MCYT baseline corpus: a bimodal biometric database. IEE Proceedings-Vision, Image and Signal Processing. 2003 Dec 1;150(6):395-401.

<a id="3">[3]</a> 
Shen C, Cai Z, Guan X, Maxion R. Performance evaluation of anomaly-detection algorithms for mouse dynamics. computers & security. 2014 Sep 1;45:156-71.

<a id="4">[4]</a> 
Tolosana R, Gismero-Trujillo J, Vera-Rodriguez R, Fierrez J, Ortega-Garcia J. MobileTouchDB: Mobile touch character database in the wild and biometric benchmark. InProceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops 2019 (pp. 0-0).

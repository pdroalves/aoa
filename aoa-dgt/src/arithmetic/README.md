# cuPoly's coprimes

cuPoly uses two set of coprime integers to build its RNS basis. The size of each of these integers depends on the context (e.g. SPOG-BFV uses 63-bits integers and SPOG-CKKS uses 63, 55, and 52-bits integers) and they must satisfy DGT's constraints to assert the existence of the k-th primitive root of i mod p:

* p \equiv 1 mod 4
* k is a power of 2
* 4k | (p-1)

There are pre-selected lists of different-size coprimes in ``src/arithmetic/coprimes`` and their decomposition in Gaussian integers..

# generate_coprimes

The python script ``src/arithmetic/generate_coprimes.py`` can be used to compute the primitive root related to each coprime and all the required inverses. The roots are queried from WolframAlpha. The outcome is a ``coprimes.cu`` file to be used by cuPoly.

The use of this script is quite simple, simply call it followed by the list of coprimes shall be compiled to ``coprimes.cu``. For instance,

```
python3 generate_coprimes.py coprimes/qis_63.json coprimes/qis_45.json coprimes/qis_48.json coprimes/qis_52.json coprimes/qis_55.json
```

will have as output the data needed to run cuPoly with 63, 55, 52, 48, and 45-bits coprimes.

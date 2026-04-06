#ifndef CKKSENCODER_H
#define CKKSENCODER_H

#include <cuda.h>
#include <newckks/cuda/htrans/common.h>
#include <newckks/arithmetic/poly_t.h>
#include <newckks/ckks/ckkscontext.h>
#include <NTL/ZZ.h>

void ckks_encode(
	CKKSContext *ctx,
	uint64_t *a,
	std::complex<double> *d_val,
	int slots,
	int empty_slots,
	uint64_t scalingfactor);

void ckks_decode(
	CKKSContext *ctx,
	std::complex<double> *d_val,
	uint64_t *a,
	int slots,
	uint64_t scalingfactor);

void rotate_slots_right(
	Context *ctx,
	poly_t *b,
	poly_t *a,
	int rotSlots);

void rotate_slots_left(
	Context *ctx,
	poly_t *b,
	poly_t *a,
	int rotSlots);

void conjugate_slots(
	Context *ctx,
	poly_t *d_a);

void fftSpecialInv(std::complex<double>* vals, const long size) ;
#endif
#ifndef CKKSCONTEXT_H
#define CKKSCONTEXT_H

#include <utility> 
#include <rapidjson/document.h>
#include <rapidjson/writer.h>
#include <rapidjson/stringbuffer.h>
using namespace rapidjson;
typedef Document json;

#include <cuComplex.h>
#include <complex>
#include <map>

#include <newckks/arithmetic/poly_t.h>
#include <newckks/ckks/cipher_t.h>
#include <newckks/cuda/htrans/common.h>
#include <newckks/ckks/ckkskeys.h>
#include <newckks/tool/context.h>
#include <string>

static std::string LN1MINUS = "Log1Minus"; ///< log(1-x)
static std::string LOGARITHM = "Logarithm"; ///< log(x)
static std::string EXPONENT  = "Exponent"; ///< exp(x)
static std::string SIN  = "Sin"; ///< sin(x)
static std::string COS  = "Cos"; ///< cos(x)
static std::string SIGMOID  = "Sigmoid"; ///< 1/1+exp(-x))

class CKKSContext : public Context{
    private:
      std::vector<Context*> alt_ctxs; // Alternative contexts at base Q
      
      std::vector<uint64_t> scalingfactor;


    public: 
      CKKSKeychain keys;

      /////////////
      // Sampler //
      /////////////


      /////////////////////////
      // Temporary variables //
      /////////////////////////
      poly_t *u, *m; // Encryption
      poly_t *e; // Encryption

      poly_t axbx1, axbx2, axax, bxbx, axmult, bxmult;
      poly_t *d2axaxQB; //Mul
      poly_t d_tildeQB[2];  //Mul
      poly_t d_tildeQ[2];  //Mul

      cipher_t caux[3]; // Auxiliar ciphertexts

      std::complex<double> *h_val; // Auxiliar for encoding
      cuDoubleComplex *d_val; // Auxiliar for encoding

      ///////////////////
      // Taylor series //
      ///////////////////
      std::map<std::string, double*> maclaurin_coeffs; ///< precomputed maclaurin coefficients

      CKKSContext() : Context(){
        assert(CUDAManager::is_init);

        //////////////
        // Pre-comp //
        //////////////
        // Auxiliar contexts
        // for(unsigned int i = 0; i < 3; i++)
        //     alt_ctxs.push_back(new Context());

        u = new poly_t;
        m = new poly_t;
        e = new poly_t;
        d2axaxQB = new poly_t;

        poly_init(this, u);
        poly_init(this, m);
        poly_init(this, e);
        poly_init(this, d2axaxQB, QBBase);
        poly_init(this, &d_tildeQB[0], QBBase);
        poly_init(this, &d_tildeQB[1], QBBase);
        poly_init(this, &d_tildeQ[0]);
        poly_init(this, &d_tildeQ[1]);
        poly_init(this, &bxbx);
        poly_init(this, &axax);
        poly_init(this, &axbx1);
        poly_init(this, &axbx2);

        for(unsigned int i = 0; i < 3; i++)
            cipher_init(this, &caux[i]);

        // Encoding stuff
        h_val = (std::complex<double>*) malloc (CUDAManager::N * sizeof(std::complex<double>));
        cudaMalloc((void**)&d_val, CUDAManager::N * sizeof(std::complex<double>));
        cudaCheckError();


        // Function approximation through Maclaurin sequences
        maclaurin_coeffs.insert(std::pair<std::string, double*>(
            EXPONENT,
            new double[11]{1,(double)1,(double)1./2,(double)1./6,(double)1./24,(double)1./120,(double)1./720,(double)1./5040,(double) 1./40320,(double)1./362880,(double)1./3628800}
            )
        );
        maclaurin_coeffs.insert(std::pair<std::string, double*>(
            SIN,
            new double[12]{0,(double)1,(double)0,(double)-1./6,(double)0,(double)1./120,(double)0,(double)-1./5040,(double)0,(double)1./362880,(double)0,(double)-1./39916800}
            )
        );
        maclaurin_coeffs.insert(std::pair<std::string, double*>(
            COS,
            new double[12]{1,(double)0,(double)-0.5,(double)1./24,(double)0,(double)-1./720,(double)0,(double)1./40320,(double)0, (double)-1./362880, (double)0, (double)1./362880}
            )
        );
        maclaurin_coeffs.insert(std::pair<std::string, double*>(
            SIGMOID,
            // new double[4]{(double)1./2,(double)0.197,(double)0,(double)-0.004}
            new double[10]{(double)1./2,(double)1./4,(double)0,(double)-1./48,(double)0,(double)1./480,(double)0,(double)-17./80640,(double)0,(double)31./1451520}
            )
        );
        // An worse approximation:
        //new double[4]{(double)0.5,(double)0.197, (double)0, -(double)0.004})
        maclaurin_coeffs.insert(std::pair<std::string, double*>(
            LN1MINUS,
            new double[21]{0, (double)-1., (double)-1./2, (double)-1./3, (double)-1./4, (double)-1./5, (double)-1./6, (double)-1./7, (double)-1./8, (double)-1./9, (double)-1./10, (double)-1./11, (double)-1./12, (double)-1./13, (double)-1./14, (double)-1./15, (double)-1./16, (double)-1./17, (double)-1./18, (double)-1./19, (double)-1./20}
            )
        );
        maclaurin_coeffs.insert(std::pair<std::string, double*>(
            LOGARITHM,
            new double[40]{0, (double)1., 0, (double)1./3, 0, (double)1./5, 0, (double)1./7, 0, (double)1./9, 0, (double)1./11, 0, (double)1./13, 0, (double)1./15, 0, (double)1./17, 0, (double)1./19, 0, (double)1./21, 0, (double)1./23, 0, (double)1./25, 0, (double)1./27, 0, (double)1./29, 0, (double)1./31, 0, (double)1./33, 0, (double)1./35, 0, (double)1./37, 0, (double)1./39}
            )
        );

        ckkskeychain_init(this, &keys);

      };

      /**
       * @brief      export all keys to a json structure
       *
       * @return     A json structure containing the sk, pk, and evk
       */
      json export_keys();

      /**
       * @brief      load keys from a json structure
       *
       * @param[in]  k     A json structure containing the sk, pk, and evk
       */
      void load_keys(const json & k);

      /**
       * @brief      Clear all the keys stored in the FV object
       */
      void clear_keys();
      /**
       * @brief      Return alternative Contexts
       * 
       * This class carries alternative contexts that can be selected by an ID. 
       * If no Context exists with a particular ID, it shall be created.
       *
       * @param[in]  id    The identifier
       *
       * @return     The alternate context.
       */
      Context* get_alt_ctx(unsigned int id){
          while(id >= alt_ctxs.size())
              alt_ctxs.push_back(new Context());
          return alt_ctxs[id];
      }

     /**
       * @brief      Synchronizes all related streams
       * 
       * Calls cudaStreamSynchronize() for all related streams
       *
       *
       * @return     .
       */ 
      void sync(){
          cudaStreamSynchronize(get_stream());
          cudaCheckError();
          sync_related();
      }

     /**
       * @brief      Synchronizes all related streams
       * 
       * Calls cudaStreamSynchronize() for all related streams
       *
       *
       * @return     .
       */ 
      void sync_related(){
          for(auto c : alt_ctxs){
              cudaStreamSynchronize(c->get_stream());
              cudaCheckError();
          }
      }
      
      void encode(poly_t *a, int64_t *scale, std::complex<double> *val, int slots, int empty_slots = 0);
      void decode(std::complex<double> *val, poly_t *a, int64_t scale, int slots);
      
      ~CKKSContext(){

        ckkskeychain_free(this, &keys);
        poly_free(this, u);
        poly_free(this, m);
        poly_free(this, e);
        poly_free(this, d2axaxQB);
        delete d2axaxQB;
        poly_free(this, &d_tildeQB[0]);
        poly_free(this, &d_tildeQB[1]);
        poly_free(this, &d_tildeQ[0]);
        poly_free(this, &d_tildeQ[1]);
        poly_free(this, &bxbx);
        poly_free(this, &axax);
        poly_free(this, &axbx1);
        poly_free(this, &axbx2);

        for(unsigned int i = 0; i < 3; i++)
            cipher_free(this, &caux[i]);

        delete u;
        delete m;
        delete e;
        for(const auto& value: alt_ctxs)
            delete value;
        alt_ctxs.clear();

        free(h_val);
        cudaFree(d_val);
        cudaCheckError();

        delete[] maclaurin_coeffs[EXPONENT];
        delete[] maclaurin_coeffs[SIN];
        delete[] maclaurin_coeffs[COS];
        delete[] maclaurin_coeffs[SIGMOID];
        delete[] maclaurin_coeffs[LN1MINUS];
        delete[] maclaurin_coeffs[LOGARITHM];
        maclaurin_coeffs.clear();
      }
};

#endif

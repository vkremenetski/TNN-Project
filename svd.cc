#include "itensor/all.h"

using namespace itensor;

int main()
    {
    //
    // SVD of matrix M
    //

    int Nrow = 4;
    int Ncol = 3;
    auto maxm = std::min(Nrow,Ncol);

    auto M = Matrix(Nrow,Ncol);
    M(0,0) = 0.435839; M(0,1) = 0.223707; M(0,2) = 0.10;
    M(1,0) = 0.435839; M(1,1) = 0.223707; M(1,2) = -0.10;
    M(2,0) = 0.223707; M(2,1) = 0.435839; M(2,2) = 0.10;
    M(3,0) = 0.223707; M(3,1) = 0.435839; M(3,2) = -0.10;
    Print(M);

    Matrix U,V;
    Vector d;
    SVD(M,U,d,V);

    Print(U);
    Print(d);
    Print(V);

    int nkeep = 2;
    auto Dtrunc = Matrix(maxm,maxm);
    for(auto j : range(nkeep))
        {
        Dtrunc(j,j) = d(j);
        }

    auto Mtrunc = U*Dtrunc*transpose(V);
    Print(Mtrunc);

    auto diff = norm(M-Mtrunc);
    auto diff2 = sqr(diff);

    printfln("|M-Mtrunc|^2 = %.2f",diff2);

    println();
    

    //
    // SVD of two-site wavefunction
    //
    
    auto s1 = Index("s1",2,Site);
    auto s2 = Index("s2",2,Site);
    auto s3 = Index("s3",2);
    auto s4 = Index("s4",2);

    auto sing = ITensor(s1,s2);
    auto prod = ITensor(s1,s2);

    //Make sing a singlet
    sing.set(s1(1),s2(2), 1./sqrt(2));
    sing.set(s1(2),s2(1),-1./sqrt(2));

    //Make prod a product state
    prod.set(s1(1),s2(2),1.);

    for(Real mix = 0; mix <= 1.; mix += 0.1)
        {
            ITensor U(s1,s3),S(s3,s4),V(s2,s4);
            auto Combo = (1-mix)*prod+mix*sing;
            Combo = Combo/norm(Combo);
            SVD(Combo,U,S,V);
            Real entropy = 0;
            for(int j = 1; j<= 2; j+=1){
                auto iv1 = IndexVal(s3,j);
                auto iv2 = IndexVal(s4,j);
                entropy += (-pow(S.real(iv1,iv2),2))*log(pow(S.real(iv1,iv2),2));
            }
            Print(entropy);

        }


    return 0;
    }

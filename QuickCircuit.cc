#include "itensor/all.h"
#include <vector>
#include <string>

using namespace itensor;


// This function should return a random two qubit unitary with the
// following indices:
//
//          ind3        ind4
//          _|___________|_
//          |             |
//          |             |
//          |_____________|
//           |           |
//          ind1        ind2
//           
ITensor RandomTwoQubitGate(Index ind1, Index ind2, Index ind3,
        Index ind4) {
    auto sTensor = randomTensor(ind1, ind2, ind3, ind4);

    // The random tensor has elements in [0, 1]. It seems preferable to have
    // them in [-1, 1] so we shift them.
    // auto shift_func = [](Real r) { return 2 * r - 1.0; };
    // sTensor.apply(shift_func);
    // The above two lines are currently commented out because they send
    // up a lot of warnings!
    // TODO: Try with and without the shift once other code is working.

    // Only the incoming indices should be included in U in order for the
    // resulting tensor to be unitary between the incoming and outgoing
    // indices.
    ITensor U(ind1,ind2),S,V;
    svd(sTensor,U,S,V);

    // We replace the index that U and S shared with the one that S and V share
    // so that we can contract U and V without S in the middle.
    Index vs = commonIndex(S, V);
    Index us = commonIndex(U, S);
    U *= delta(us, vs);

    // We return U*V in order to have a unitary that goes from the incoming
    // to the outgoing indices.
    return U * V;
}


// This function should return a two qubit unitary that acts as
// the identity on the ind1, ind2 space.
//
//          ind3        ind4
//           |           |
//           |           | 
//           |           | 
//           |           | 
//           |           |
//          ind1        ind2
//           
ITensor IdentityTwoQubitGate(Index ind1, Index ind2, Index ind3,
        Index ind4) {

    auto Identity = ITensor(ind1, ind2, ind3, ind4);
    for(int i = 1; i<=2; i += 1)
    {
        for(int j = 1; j <= 2; j += 1) {
            Identity.set(ind1(i), ind2(j), ind3(i), ind4(j), 1);
        }
    }
    return Identity;
}
/* This function returns a gate for swapping the qubits at indices 
   ind 1 and ind 2.
            ind3        ind4
             |           |
                |     |
                   ||  
                |     |
             |           |
            ind1        ind2
*/
ITensor SwapGate(Index ind1, Index ind2, Index ind3, Index ind4){
    return delta(ind1, ind4)*delta(ind2, ind3);
}

// SimpleNetwork contains all of the tensors in our model, and is 
// able to optimize the model for a given Hamiltonian.
class SimpleNetwork {
public:
    // The constructor for the class. It is responsible for creating
    // all of the Index objects and randomly initializing all of the
    // two qubit gate Tensors.
    SimpleNetwork(int circuit_depth, int num_qubits, bool is_periodic) {
        // The circuit depth defines how many layers we have.
        circuit_depth_ = circuit_depth;
        
        // The number of qubits should be either 4, 8, or 16.
        num_qubits_ = num_qubits;

        // If the network is not supposed to be periodic we will still
        // create two qubit gates that overlap from the last qubit
        // to the first qubit, but we will always leave them to 
        // be the identity. (This is just to make the code easier to
        // write.)
        is_periodic_ = is_periodic;

        // We resize our containers ahead of time so that we
        // can use the assignment operators easily.


        my_indices_.resize(circuit_depth + 1);
        my_gates_.resize(circuit_depth);
        update_.resize(circuit_depth);
        for(int i = 0; i < circuit_depth; i++) {
            my_indices_[i].resize(num_qubits);
            my_gates_[i].resize(num_qubits / 2);
            update_[i].resize(num_qubits/2);
        }
        my_indices_[circuit_depth].resize(num_qubits);

        // We begin by creating all of the Index objects that we will
        // need. Note that x and y label coordinates on the tensor 
        // network diagram.
        for(int y = 0; y <= circuit_depth; y++)
        for(int x = 0; x < num_qubits; x++) {
            std::string name_string;
            name_string += "index (";
            name_string += std::to_string(x);
            name_string += ",";
            name_string += std::to_string(y);
            name_string += ")";
            my_indices_[y][x] = Index(name_string, 2);
        }
        
        // Now we create all of the gates that we'll need.
        for(int y = 0; y < circuit_depth; y++)
        for(int x = 0; x < num_qubits; x++) {
            // We only want to create a two qubit gate at every other
            // pair of qubits. We choose to start at qubit 1 on even
            // numbered layers and qubit 0 on odd numbered layers by
            // skipping in other cases.
            if ((x + y) % 2 == 0)
                continue;

            // We pick out our four indices:
            Index ind1 = my_indices_[y][x];
            Index ind2 = my_indices_[y][(x + 1) % num_qubits];
            Index ind3 = my_indices_[y + 1][x];
            Index ind4 = my_indices_[y + 1][(x + 1) % num_qubits];

            // If we are not periodic and are creating a tensor that
            // overlaps from the last qubit to the first qubit we should
            // set it to be the identity, otherwise we initialize
            // with a random unitary.
            ITensor new_gate;
            if (is_periodic || x < num_qubits - 1){
                new_gate =  RandomTwoQubitGate(ind1,
                        ind2, ind3, ind4);
                update_[y][x/2] = true;
            }
            else{
                new_gate = IdentityTwoQubitGate(ind1,
                        ind2, ind3, ind4);
                update_[y][x/2] = false;
            }

            my_gates_[y][x/2] = new_gate;
        }
    }
    // This function starts at the "top" of the circuit (with ancilla qubits
    // in the 0 state) and applies every gate down through the level of the
    // circuit specified by level.
    //
    // XX (Within a level, it applies the gates in order up to (and including)
    // the gate at gate_position) XX.

    // VLADIMIR: No longer including the gate at gate_position. I have modified
    // subsequent code accordingly. Furthermore, in the code I assume that if "level"
    // is the top level in the circuit, the corresponding ancilla qubit will not
    // be applied to the target gate, nor to the gates beyond it (if there are any). 
    ITensor
    contractDown(int level, int gate_position)  {
        int  currentLevel = circuit_depth_ - 1;
        Index i1 = my_indices_[currentLevel+1][0];
        ITensor env = ITensor(i1);
        env.set(i1(1),1);
        for(int x = 1; x < num_qubits_; x++) {
            Index anIndex = my_indices_[currentLevel+1][x];
            ITensor ancilla = ITensor(anIndex);
            ancilla.set(anIndex(1),1);
            env *= ancilla;
        }

        //VLADIMIR: Here, the variable "bound" determines how far along a level we go.
        //I simply change bound when current level == target level, to avoid writing
        // a separate but near identical case for that. Likewise for ContractUp.
        int bound = num_qubits_/2;
        while(currentLevel >= level) {
            if(currentLevel==level){bound = gate_position;}
            for(int x = 0; x < bound; x++) {
                env *= my_gates_[currentLevel][x];
            }
            currentLevel--;
        }

        return env;
    }

    // This function starts at the "bottom" of the circuit by contracting the
    // entire conjugate wavefunction. It then applies the operator ham
    // (usually but not necessarily a Hamiltonian) to the resulting conjugate
    // wavefunction and contracts the entire wavefunction up through the level
    // of the circuit specified by level.
    //
    // Within a level, it applies the gates in reverse order, up to (but
    // not including) the gate at gate_position.
    ITensor
    contractUp(int level, int gate_position, const MPO & ham)  {
        // We get the conjugate wavefunction.
        ITensor to_return = contractDown(0, num_qubits_/2);
        to_return = dag(to_return.prime());

        // We apply the MPO operator.
        for (int i = 0; i < num_qubits_; i++) {
            auto mpo_index = findtype(ham.A(i + 1), Site).noprime();

            // We set the indices up so our existing tensor will
            // contract with this piece of the MPO.
            to_return *= delta(prime(my_indices_[0][i]), prime(mpo_index));

            // We contract with an MPO tensor (note that they are 1 indexed
            // whereas we are zero-indexing our vectors).
            to_return *= ham.A(i + 1);

            // And we switch back to our indices.
            to_return *= delta(my_indices_[0][i], mpo_index);
        }
        int currentLevel = 0;
        int bound = -1;
        while(currentLevel<= level){
            if(level==currentLevel){bound = gate_position;}
            for(int i=num_qubits_/2 - 1; i > bound; i--){
                to_return *= my_gates_[currentLevel][i];
            }
            currentLevel++;
        }

        // At this point, we have contracted the entire conjugate wavefunction
        // with the MPO. Now we need to contract up to the proper gate in
        // our non-conjugated wavefunction network.
        return to_return;
    }

    // This function evaluates the expectation value of an operator.
    Real
    expectationValue(const MPO & ham) {
        return (contractUp(0, 0, ham) * contractDown(0, 1)).real();
    }

    // This function finds the environment of a particular tensor.
    ITensor
    getEnvironment(int level, int gate_position, const MPO & ham) {
        return contractUp(level, gate_position, ham) * contractDown(level,
                gate_position);
    }

    // This function updates a particular gate given that the others (and
    // its conjugate) are frozen.
    void
    updateGate(int level, int gate_position, const MPO & ham) {
        auto enviro = getEnvironment(level, gate_position, ham);
        auto gate = getGate(level, gate_position);
        int x_pos = 2 * gate_position + (1 + level) % 2;
        ITensor U(my_indices_[level+1][x_pos],my_indices_[level+1][(x_pos + 1)%num_qubits_]), S, V;
        svd(enviro, U,S,V);
        Index sv = commonIndex(S,V);
        Index us = commonIndex(U,S);
        U *= delta(us, sv);
        my_gates_[level][gate_position] = dag(U)*dag(V);
    }
    ITensor
    getGate(int level, int gate_position){
        return my_gates_[level][gate_position];
    }

    /* This function loops over each gate in the network and updates 
       the non-fixed gates (i.e. not identity or swap).
    */
    void
    optimizationStep(const MPO & ham) {
        for(int i = 0; i< circuit_depth_; i++) {
            for(int j = 0; j<num_qubits_/2; j++) {
                if(update_[i][j]){
                    updateGate(i,j, ham);
                }
            }
        } 
    }
// These variables are labeled protected so that they're accessible
// from the subclasses.
protected:
    // my_gates_ contains all of the two qubit gates in the model.
    // The first index should access the "level" of the circuit, 
    // and the second accesses left to right in the tensor network
    // diagram.
    std::vector< std::vector<ITensor> > my_gates_;

    // my_indices contains all of the Index objects that are
    // used by the gate Tensors in the model.
    // The first index should access the "level" of the circuit,
    
    // and the second accesses the qubits left to right in the tensor
    // network diagram.
    std::vector< std::vector<Index> > my_indices_;

    //update_ contains booleans indicating whether the gates
    // in my_gates_ at the same indices should be updated.
    std::vector< std::vector<bool> > update_;

private:
    // Set to True if model is periodic, else False.
    bool is_periodic_;

    // The total depth of the circuit.
    int circuit_depth_;

    // The number of qubits.
    int num_qubits_;
};
class meraNetwork : public SimpleNetwork{
    public:
        meraNetwork(std::vector< std::vector<int> > layout, bool is_periodic): 
        SimpleNetwork(layout.size(), layout[0].size()*2, is_periodic){
            layout_ = layout;
            int ng = layout[0].size(); //number of gates per level
            int l = layout.size(); //level count of the circuit
            for(int i = 0; i < l; i++){
                for(int j =0; j < ng; j++){
                    int g = layout_[i][j];
                    if(g!=1){
                        update_[i][j] = false;
                        Index ind1 = my_indices_[i][2*j];
                        Index ind2 = my_indices_[i][(2*j+1) % (2*ng)];
                        Index ind3 = my_indices_[i+1][2*j];
                        Index ind4 = my_indices_[i+1][(2*j+1) %(2*ng)];
                        printfln("Hey, this is level", j);
                        if(g == 0) {
                            my_gates_[i][j] = IdentityTwoQubitGate(ind1,ind2,ind3,ind4);
                        } else {
                            my_gates_[i][j] = IdentityTwoQubitGate(ind2,ind1,ind3,ind4);
                        }
                        
                    }
                }
            }
        }
    private:
        //layout_ keeps track of the desired layout of the circuit.
        // 0 stands for "identity", 1 for "random unitary", and 2 for SWAP.
        std::vector< std::vector<int> > layout_;
};

Real
sanityChecksForSimpleNetwork() {
    auto network_1 = SimpleNetwork(3, 4, true);
    auto network_2 = SimpleNetwork(9,8,false);
    int N = 8;
    auto sites = SpinHalf(N);
    auto psi = MPS(sites);

    auto ampo = AutoMPO(sites);
    for(int j = 1; j < N; ++j) {
        ampo += "Sz",j,"Sz",j+1;
        ampo += 0.5,"S+",j,"S-",j+1;
        ampo += 0.5,"S-",j,"S+",j+1;
    }

    auto H = MPO(ampo);
    //Regular periodic Mera for 8 qubits
    std::vector< std::vector<int> > layout8q;
    layout8q.resize(5);
    for(int i = 0; i < 5; i++) {
        layout8q[i].resize(4);
        if (i <= 1)
        {
            for(int j = 0; j<4; j++){
                layout8q[i][j] = 1;
            }
        }
    }
    layout8q[4][1] = 1;
    layout8q[3][1] = 1;
    layout8q[3][2] = 1;
    layout8q[2][0] = 2;
    layout8q[2][1] = 1;
    layout8q[2][2] = 2;
    auto state = meraNetwork(layout8q,true);
    printfln("Old Expectation Value", network_2.expectationValue(H));
    Real e = network_2.expectationValue(H);
    Index i1 = Index("s1",2);
    Index i2 = Index("s2",2);
    Index i3 = prime(i1);
    Index i4 = prime(i2);
    //auto g = SwapGate(i1,i2,i3,i4);
    /*for(int i = 0; i<50; i++){
        network_2.optimizationStep(H);
        e = network_2.expectationValue(H);
        printfln("Energy Estimate ", e);
    }
    /*auto sweeps = Sweeps(5)
    sweeps.maxm() = 50,50,100,100,200;300;
    sweeps.cutoff() = 1E-9;
    Real e = dmrg(psi,H,sweeps,"Quiet");
    /*Real e = network_2.expectationValue(H);*/
    return  e;


}

int main(int argc, char* argv[]) {
    Real e = sanityChecksForSimpleNetwork();
    printfln("Updated Expectation Value ", e);
}

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

    // TODO: Implement this method!
    return ITensor();
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

    // TODO: Implement this method!
    return ITensor();
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
        for(int i = 0; i < circuit_depth; i++) {
            my_indices_[i].resize(num_qubits);
            my_gates_[i].resize(num_qubits / 2);
        }
        my_indices_[circuit_depth].resize(num_qubits);

        // We begin by creating all of the Index objects that we will
        // need. Note that x and y label coordinates on the tensor 
        // network diagram.
        for(int y = 0; y <= circuit_depth; y++)
        for(int x = 0; x < num_qubits; x++) {
            std::string name_string;
            name_string += "index (";
            name_string += x;
            name_string += ",";
            name_string += y;
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
            if (is_periodic || x < num_qubits - 1)
                new_gate =  RandomTwoQubitGate(ind1,
                        ind2, ind3, ind4);
            else
                new_gate = IdentityTwoQubitGate(ind1,
                        ind2, ind3, ind4);

            my_gates_[y][x/2] = new_gate;
        }
    }

private:
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


    // Set to True if model is periodic, else False.
    bool is_periodic_;

    // The total depth of the circuit.
    int circuit_depth_;

    // The number of qubits.
    int num_qubits_;
};


int main(int argc, char* argv[]) {

    auto network_1 = SimpleNetwork(3, 4, true);
    auto network_2 = SimpleNetwork(5, 8, false);


    return 0;
}

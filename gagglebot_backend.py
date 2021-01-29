import random

#Implementation of the NEAT algorithm by Zachary Smith using the following sources
#https://www.youtube.com/watch?v=VMQOa4-rVxE&list=WL&index=115
#http://nn.cs.utexas.edu/downloads/papers/stanley.ec02.pdf
#https://en.wikipedia.org/wiki/Neuroevolution_of_augmenting_topologies#HyperNEAT
#
#
#The goal of this project is to develop a submission for the Fighting AI Competition, which is helf in order to drive development of general fighting game AIs
#Because my goal is to develop an algorithm that can be generally applied to fighting games, I assume as little about the environment as possible
#In that regard however, some concessions were made as I am still very new to implementing these kinds of things on my own, and it is my first experience with the NEAT
#algorithm. 
#The things I hard-code into the bot about the environment are as follows:
# the bot's longest range "normal", aka its longest range non-special attack
# the throw range of the bot
#
#These two pieces of information will be explained later as they are relevant, however for now they are inputs for the AI
#
#I would propose that a more general AI would figure these thigns out by itself as well in real-time, however to save myself time and energy to focus on the NEAT, I chose
#not to pursue doing that myself
#
#I chose to do a non-visual based AI for a similar reason, I wanted to save myself some time and be able to focus on the NEAT itself
#
#For future submissions I will likely make the needed changes in those two aspects
#
#
#I chose to use the NEAT (NeuroEvolution of Augmenting Topologies) algorithm for this project for a few reasons:
# 1) I really enjoy genetic algorithms, and this helps me learn more about them
# 2) I believe that with the right scope, NEAT can emulate how a human would play and adapt in the middle of a set in fighting games
# 3) It allows me to assume the least about the environment (at least out of the selection of other ideas I had)

#Connections specify two nodes, one input, one output
#It also specifies an innovation number, used to resolve competing connections
#Finally, a connection can be enabled or disabled
class Connection:

    def __init__(self, i, o, innovation, enabled = True):
        self._i = i
        self._o = o
        self._innovation = innovation
        self._enabled = enabled

        #When a connection is first created between two nodes, add the output node to the input node's output list
        self._i.add_output(self._o)

    #Toggles whether the connection is enabled or not
    def toggle(self):
        if self._enabled:
            self._enabled = False
            self._i.remove_output(self._o)
        else:
            self._enabled = True
            self._i.add_output(self._o)
        return

    #Returns a tuple of the input and output node
    def get_connection(self):
        return (self._i, self._o)

    #Returns the innovation number associated with this connection
    def get_innovation(self):
        return self._innovation

    #Returns whether the connection is enabled or not
    def is_enabled(self):
        return self._enabled

#Neurons hold the following information:
#output: A set of neurons this neuron is connected to
#layer: A value indicating the layer the node belongs to
class Neuron:

    def __init__(self, layer = 0):
        self._outputs = set()
        self._layer = 0
        #Used as a solution to a problem with distance setting
        self._temp_layer = 0

    #Functions to modify and get the attributes of the neuron

    def add_output(self, output):
        self._outputs.add(output)
        return

    def remove_output(self, output):
        self._outputs.remove(output)
        return

    def set_layer(self, layer):
        self._layer = layer
        return

    def set_temp_layer(self, layer):
        self._temp_layer = layer

    def get_outputs(self):
        return self._outputs

    def get_layer(self):
        return self._layer

    def get_temp_layer(self):
        return self._temp_layer

    def push(self):
        outputs = ""
        for neuron in self._outputs:
            output = neuron.push()
            if output != None:
                outputs = outputs + output + "+"
        return outputs


#A neuron whose output is a simple key for a button
class Output_Neuron(Neuron):

    def __init__(self, command, layer = 0):
        super().__init__()
        self._command = command

    def push(self):
        return self._command

    def get_outputs(self):
        return None

class Network:

    def __init__(self):
        self._neurons = []
        self._connections = set()
        self._innovation = 0
        self._inputs = 0

    #inputs is an integer corresponding to the number of inputs
    #outputs is a list of strings corresponding to each possible button press
    def build_default(self, inputs, outputs):
        self._inputs = inputs
        for i in range(len(inputs)):
            self._neurons.append(Neuron())

        for i in range(len(outputs)):
            self._neurons.append(Output_Neuron(outputs[i], 1))
        #In a default net, the innovation number of the network is 0 since there are no connections
        return

    #Builds a network from a set of connections passed to this method
    #Layering is performed after/before mutation depending on what mutation is done
    def build_from_connections(self, connections):
        #Use a set here to eliminate duplicates
        neurons_to_add = set()
        for connection in connections:
            if connection.is_enabled():
                neurons = connection.get_connection()
                neurons_to_add.add(neurons[0])
                neurons_to_add.add(neurons[1])
        #With all the relevant neurons added, simply set the connections attribute to the passed set
        self._connections = connections
        return

    def get_enabled_connection(self):
        to_return = []
        for connection in self._connections:
            if connection.is_enabled():
                to_return.append(connection)
        return to_return

    #Keep the neurons sorted by layer
    def layer(self):
        #Check and set all node's layer values based on distance from inputs
        for i in range(self._inputs):
            input_neuron = self._neurons[i]
            set_distance(input_neuron)
        #Now iterate through all the neurons, and set their layers to their temp layers
        for neuron in self._neurons:
            neuron.set_layer(neuron.get_temp_layer())
            #Now set the temp layer to an arbitrarily high number so that it "resets", (max_neurons + 1?)
            neuron.set_temp_layer(1000)
        #Finally, sort the list based on the layers
        self._neurons.sort(lambda neuron: neuron.get_layer())
        return

    #Give input, get output
    def run(self, inputs):
        for i in range(self._inputs):
            if inputs[i] == 0:
                continue
            output = self._neurons[i].push()
        return output

    #Used for mutation
    def add_neuron(self, neuron):
        self._neurons.append(neuron)
        return

    #Used for mutation
    def add_connection(self, connection):
        #Any new connection created will automatically have a higher innovation number
        self._innovation = connection.get_innovation()
        self._connections.add(connection)
        return

    #Returns the entire set of neurons
    def get_neurons(self):
        return self._neurons

    #Returns the number of neurons in subsequent layers
    def get_num_neurons_after(self, layer):
        counter = 0
        for neuron in self._neurons:
            if neuron.get_layer() > layer:
                counter += 1
        return counter

    def get_neurons_after(self, layer):
        neurons = []
        for neuron in self._neurons:
            if neuron.get_layer() > layer:
                neurons.append(neuron)
        return neurons

    def copy(self):
        neurons = self._neurons.copy()
        connections = self._connections.copy()
        innovation = self._innovation
        inputs = self._inputs

        net = Network()
        net._neurons = neurons
        net._connections = connections
        net._innovation = innovation
        net._inputs = inputs
        return net


#PROBLEM: HOW TO SET ONLY TO LOWEST DISCOVERED DISTANCE IN THE CURRENT DISTANCE CHECK, I.E IGNORE DISTANCE FROM PREVIOUS GENERATION
#SOLUTION: HACKY BUT JUST USE A TEMP_DISTANCE VARIABLE
def set_distance(neuron, distance = 0):
    #Check against the temp layer, to set the value later
    #Largest distance found is the layer, so check if the temp layer is less than the current distance
    if neuron.get_temp_layer() < distance:
        neuron.set_temp_layer(distance)
    if neuron.get_outputs() != None:
        distance += 1
        for output in neuron.get_outputs():
            set_distance(output, distance)
    return


#For mutation: 
#layer the network BEFORE connection mutations, and no need to layer it again afterwards
#layer the network AFTER neuron mutations, no need to layer it before
#To only layer ONCE, perform Neuron mutation first, THEN connection mutation

#Neuron mutation: Add a new neuron between an existing enabled connection
#Randomly select a connection, and insert a neuron
def neuron_mutation(network):
    #Get a random selection from the enabled connections
    connections = network.get_enabled_connections()
    choice = random.randint(0, len(connections) - 1)
    chosen_connection = connections[choice]

    #Toggle the chosen connection
    chosen_connection.toggle()

    #With the random connection chosen
    #First, create a new neuron, layer does not matter here as the net will be relayered anyways
    neuron = Neuron()

    #Next take the input and output neurons from the chosen connection
    neurons = chosen_connection.get_connection()
    input_neuron = neurons[0]
    output_neuron = neurons[1]

    #Create a connection between the input and new neuron, then add it
    in_connection = Connection(input_neuron, neuron, int(network.get_innovation) + 1)
    network.add_connection(in_connection)
    #Create a connection between the new neuron and the output, then add it
    out_connection = Connection(neuron, output_neuron, int(network.get_innovation) + 1)
    network.add_connection(out_connection)
    #Add the new neuron
    network.add_neuron(neuron)
    #Mutation complete, relayer the network
    network.layer()
    return

#Connection mutation: Add a new connection between two previously unconnected neurons
#Randomly select one neuron, then randomly select another that it is not connected to
def mutate_connection(network):
    #First, select a neuron
    neurons = network.get_neurons()
    choice = random.randint(0, len(neurons) - 1)

    neuron = neurons[choice]

    tries = 0
    #Next, check to ensure the chosen neuron has neurons it is not connected to
    while(network.get_num_neurons_after(neuron.get_layer()) == 0):
        choice = random.randint(0, len(neurons) - 1)
        neuron = neurons[choice]
        tries += 1
        #Have a condition for too many tries
        if tries >= len(neurons):
            print("All connections filled")
            return None

    #Now, select a neuron that it is not connected to from the neurons in subsequent layers
    neurons = network.get_neurons_after(neuron.get_layer())
    for output in neuron.get_outputs():
        neurons.remove(output)

    choice = random.randint(0, len(neurons) - 1)
    output = neurons[choice]

    #Now create the connection
    connection = Connection(neuron, output, int(network.get_innovation()) + 1)

    #Add the connection
    network.add_connection(connection)
    return

#Crossover
#Take two parent genes, and line up their connection genes
#Any connections with the same innovation number, randomly select between the two
#Any disjoint connections, add them automatically
def crossover(parent1, parent2):
    #Unwrap from Genes
    parent1 = parent1.get_network()
    parent2 = parent2.get_network()
    #Get the list of connections from each
    #p1c = parent1 connections
    #p2c = parent2 connections
    p1c = parent1.get_enabled_connections()
    p2c = parent2.get_enabled_connections()

    #Create a set of connections for the child
    child_connections = set()

    #Now find the longest list of parent connections, and place it into p1c
    if len(p1c) < (p2c):
        temp = p1c
        p1c = p2c
        p2c = temp
    #Iterate over the longest list, and see if the innovation number is found in the other parent's list. if not, add the connection, otherwise random select
    for connection in p1c:
        competition = False
        innovation = connection.get_innovation()
        for p2_connection in p2c:
            if innovation == p2_connection.get_innovation():
                #if found, get a float between 0 and 1
                choice = random.random()
                if choice > 0.5:
                    child_connections.add(connection)
                else:
                    child_connections.add(p2_connection)
                competition = True
                #Now remove the two connections from each as they do not need to be checked again
                p1c.remove(connection)
                p2c.remove(p2_connection)
                break
        if not competition:
            child_connections.add(connection)
    #Now, for any remaining connections in p2c, they must be disjoint, so add them
    for connection in p2c:
        child_connections.add(connection)

    #New list of connections made, create a new network and return it
    new_net = Network()
    new_net.build_from_connections(child_connections)

    #Rewrap into a gene
    return Gene(new_net)

#The encoding of a gene so it can hold the information relevant for its fitness to be assessed
class Gene:

    def __init__(self, network):
        self._network = network
        self._fitness = 0
        #Allow the following three values to be accessed outside the class to keep the class brief
        self.hits_landed = 0
        self.times_hit = 0
        self.rounds_won = 0

    def calc_fitness(self):
        #Fitness function theory goes as follows:
        #Reward the number of hits landed, as that is how a round is won, allowing the algorithm to converge into winning a round
        #Penalize the number of times the bot was hit, as that is how it will lose. Penalize getting hit at about 1/2 of the reward for landing hits, so the bot
        #does not just run away
        #Highly reward for each round won, as that is how it will win a game, and there will be at most 2, while there can be many many hits
        #THerefore, reward it at about 10 * reward for a hit
        reward = 0
        reward += 2 * self.hits_landed
        reward -= self.times_hit
        reward += 20 * self.rounds_won

        self._fitness = reward
        return

    def mutate_connection(self):
        self._network.mutate_connection()
        return

    def mutate_neuron(self):
        self._network.mutate_neuron()
        return

    def get_fitness(self):
        return self._fitness

    def copy(self):
        gene = Gene(self._network.copy())
        return gene

    def get_network(self):
        return self._network

#Finally, the class that will load the bot itself to put all this together and perform learning
class Bot:

        def __init__(self):
            self._population = []
            self._used_genes = []
            self._pop_max = 10
            #Create a default net. Later on have it load in the latest optimal net
            #The inputs will be 3 distance buckets, 4 buckets for opponent state, and 2 buckets corresponding to whether the opponent or itself has enough meter for a special
            #Therefore in total, there are 9 inputs
            #For outputs, its simply the set of buttons
            self._outputs = ["A", "B", "C", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
            self._num_inputs = 9

            #20% chance of mutating a new node
            self._neuron_chance = 0.2

            #40% chance of mutating a new connection
            self._connection_chance = 0.4

        #Initializes the bot to run from scratch
        def from_scratch(self):
            default_net = Network()
            default_net.build_default(self._num_inputs, self._outputs)
            default_gene = Gene(default_net)
            self._population.append(default_gene)
            #Fill out the rest of the initial population with these default nets
            while(len(self._population) < self._pop_max):
                self._population.append(default_gene.copy())

            #Next, mutate a connection in each network so that it can run initially
            for gene in self._population:
                gene.mutate_connection()
            return

        def run(self):
            #Return the next gene from the population
            return self._population.pop()

        def record(self, gene):
            self._used_genes.append(gene)
            return

        def roulette(self):
            #Sum all the fitnesses
            fitness_sum = 0
            for gene in self._used_genes:
                fitness_sum += gene.get_fitness()

            #Make a random choice between 0 and the fitness sum
            choice = random.randint(0, fitness_sum)
            
            counter = 0
            for gene in self._used_genes:
                counter += gene.get_fitness()
                if counter >= choice:
                    parent = gene

            return parent

        def next_gen(self):
            new_pop = []
            #Perform the generation process. In each generation have half the population be children, and the other half be elite from previous
            #Because opponent will often act differently in each round, input/output is not set in stone, meaning that top % from previous may be bottom % of next

            #Sort the used genes by their fitness
            self._used_genes.sort(lambda gene: gene.get_fitness())

            #Next generate 5 children with roulette selected parents
            for i in range(5):
                parent1 = self.roulette()
                parent2 = self.roulette()
                child = crossover(parent1, parent2)

                #Then check if the child will be mutated
                #First neuron mutation
                mutate = random.random()
                if mutate <= self._neuron_chance:
                    child.mutate_neuron()
                #Then connection mutation
                mutate = random.random()
                if mutate <= self._connection_chance:
                    child.mutate_connection()
                #Finally, append to the new pop
                new_pop.append(child)

            #After 5 children are generated, take the top 5 fittest
            for i in range(-1, -6, -1):
                new_pop.append(self._used_genes[i])
            #Now set the population to = the new pop
            self._population = new_pop
            return
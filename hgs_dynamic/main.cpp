#include <time.h>
#include <iostream>

#include "Genetic.h"
#include "commandline.h"
#include "LocalSearch.h"
#include "Split.h"
#include "Params.h"
#include "Population.h"
#include "Individual.h"

// Main class of the algorithm. Used to read from the parameters from the command line,
// create the structures and initial population, and run the hybrid genetic search
int main(int argc, char* argv[])
{
	try
	{
		// Reading the arguments of the program
		CommandLine commandline(argc, argv);


		Params params(commandline);

		if ( params.config.verbose == true )
		{
			// Reading the data file and initializing some data structures
			std::cout << "----- READING DATA SET FROM: " << params.config.pathInstance << std::endl;
		}

		// Creating the Split and Local Search structures
		Split split(&params);
		LocalSearch localSearch(&params);

		// Initial population		
		if ( params.config.verbose == true )
		{
			std::cout << "----- INSTANCE LOADED WITH " << params.nbClients << " CLIENTS AND " << params.nbVehicles << " VEHICLES" << std::endl;
			std::cout << "----- BUILDING INITIAL POPULATION" << std::endl;
		}
		Population population(&params, &split, &localSearch);
		
		// Genetic algorithm
		if ( params.config.verbose == true )
			std::cout << "----- STARTING GENETIC ALGORITHM, nbIter: " << params.config.nbIter << std::endl;
		Genetic solver(&params, &split, &population, &localSearch);
		solver.run(params.config.nbIter, params.config.timeLimit);

		if ( params.config.verbose == true )
			std::cout << "----- GENETIC ALGORITHM FINISHED, TIME SPENT: " << params.getTimeElapsedSeconds() << std::endl;

		// Export the best solution, if it exist
		if (population.getBestFound() != nullptr)
		{
			population.getBestFound()->printCVRPLibFormat();
			population.getBestFound()->exportCVRPLibFormat(params.config.pathSolution);
			//population.exportSearchProgress(commandline.config.pathSolution + ".PG.csv", commandline.config.pathInstance, commandline.config.seed);
			if (params.config.pathBKS != "")
			{
				population.exportBKS(params.config.pathBKS);
			}
		}
	}

	// Catch exceptions
	catch (const std::string& e)
	{ 
		std::cout << "EXCEPTION | " << e << std::endl;
	}
	catch (const std::exception& e)
	{ 
		std::cout << "EXCEPTION | " << e.what() << std::endl; 
	}

	// Return 0 if the program execution was successfull
	return 0;
}

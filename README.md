********************************************************************************* 
                         SHORT DESCRIPTION

 This is an artificial neural network program for classification tasks and it is based on 
 gradient descent and error back propagation algorithms. This program is also a proposal to overcome the 
 problem of the local minimum. Namely, by monitoring the loss function, the local minimum is determined
 after which the percentage of successfully reproduced samples is checked. If it is >95% on both
 sets (training and validation), the global maximum has been reached. Otherwise, a small offset is
 added to the neural weights and biases, pushing the network out of the way local minimum.

********************************************************************************
                          BEFORE START 

  The program is written in C# using the Visual Studio 2022 IDE.

  It is necessary to install the namespace, System.Runtime.InteropServices in the following steps: 

  1. Right click on the project name in Solution Explorer. 
  2. Click on ManageNuGet Packages. 
  3. Select the Browse option and type in the field below System.Runtime.InteropServices. 
  4. In the list below, select System.Runtime.InteropServices.
  5. Click Install. 

  Also, It is necessary the namespace, Microsoft.Office.Interop.Excel. Follow the steps: 

  1. Right click on the project name in Solution Explorer. 
  2. Click on the Add option. 
  3. Click on the COM Reference option.
  4. Select Microsoft Excel 15.0 (or other) Object Library. 
  5. Click OK.   
  
*********************************************************************************

  The software comes with a folder under the title Input data, with prepared 
  databases for testing it.  It is necessary to set the path to this folder in the
  source code, as shown below:

    string path = @"C:\Input_data\iris_set.xlsx";

  This path can be changed according to the user's needs.

  All data are compiled into Excel tables, so that the last columns 
  represent the categories for the classification of input data. 
  For example one row in iris_set.xsl is: 

   5.1	 3.5	1.4	0.2	1	0	0

 This means that there are three categories for classification,  which 
 are set by a variable in the source code: 

   int num_class = 3 ;

 so in the case of other bases, it is necessary to change the value of this variable.
 Also, there is a possibility to center the input data (zero mean and divided by its standard deviation) 
 by setting a variable: 

   int student = 1; 

 and the input vector for the simulation will be automatically centered as well. 
 At the end of the main function, examples of input vectors for simulation are given. 
 It is enough to remove the comment in front of the given vector. For example for a wine set:

  //double[] test_input = { 0.8, 1, 1, 1, 0.5, 1, 0.8, 1,0.6};

 it is enough to remove two backslashes, //. But in that case, it is necessary to change the 
 number of categories for this database with, int num_class = 2, as well as the path of the file:

   string path = @"C:\Input_data\wine_set.xlsx". 

 It is necessary to put a comment with two backslashes, //, in front of the previously
 used vector. If it was an iris set, it will be: //double[] test_input = {5, 2, 3.5, 1}, etc.

********************************************************************************

   



  

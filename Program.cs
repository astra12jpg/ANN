using System;
using System.Data;
using System.Collections;
using System.Collections.Generic;
using Microsoft.Office.Interop.Excel;
using System.Runtime.InteropServices;
using Excel = Microsoft.Office.Interop.Excel;
using System.Diagnostics;
using System.Threading;


namespace Artifical_Network
{
    class Program
    {
        static void Main(string[] args)
        {
            var timewatch = new Stopwatch();
            timewatch.Start();
            Thread.Sleep(100);

            /******* NETWORK PARAMETERS *********/
            int Epochs = 2000;
            double alpha = 0.1;
            double error_limit = 0.0001;
            int high = 1, low = 0;
            double training_ratio = 0.75;
            int[] config = { 10 };
            /*******************************/

            /*******THE ARRAY LISTS **********/
            ArrayList st = new ArrayList();
            ArrayList net_input = new ArrayList();
            ArrayList net_enter = new ArrayList();
            ArrayList results = new ArrayList();
            /*****************************/

            /*********NET OBJECT************/
            Network net = new Network();
            /*******************************/

            /*****DATA IMPORT AND PROCESSING ******/
            string path = @"C:\Input_data\iris_set.xlsx";
            double[][] allData = net.import_file(path);
            int num_class = 3;
            int xlength = allData[0].Length - num_class;
            int student = 0;
            st = net.split_test(allData, num_class, training_ratio, student);
            net_input = net.data_proces(st, xlength, num_class, high, low);
            /****************************************/

            /*****INITIALIZATION AND TRAINING NETWORK******/
            net_enter = net.network_init(config, xlength, num_class);
            results = net.extend_backprop(config, net_input, net_enter,
                Epochs, alpha, error_limit, high, low);
            /******************************************/

            /********NETWORK PERFORMANCE ****/
            net.stats(st, results, config);
            /*****************************************/

            /******SIMULATE NETWORK***********/
            double maxx = (double)net_input[6];
            double minix = (double)net_input[7];
            double[][,] weights = (double[][,])results[0];
            double[][] bias = (double[][])results[2];
            double[] test_input = { 5, 2, 3.5, 1 }; double[] original_vector = { 0, 1, 0 }; if (student == 1) { test_input = net.studentize1D(test_input);} // iris_set
            //double[] test_input = { 0.8, 1,	1,	1,	0.5, 1,	0.8, 1,	0.6}; double[] original_vector = { 0, 1 }; if(student==1){test_input = net.studentize1D(test_input);}// cancer_set
           //double[] test_input = {6.6,0.2, 0.38, 7.9, 0.052, 30, 145, 0.9947, 3.32, 0.56, 11}; double[] original_vector = { 0, 1 }; if (student==1) {test_input = net.studentize1D(test_input);}// wine_set

            Console.WriteLine("***************************************\n");
            double[] test_resp = net.simulate(weights, bias, test_input,
               config, num_class, high, low, maxx, minix);
            Console.WriteLine("\t Network response \t Original vector");
            for (int i = 0; i < num_class; i++)
            {
                Console.WriteLine("\t {0} \t\t\t  {1}",test_resp[i], original_vector[i]);
            }
            /*******************************/
            timewatch.Stop();
            Console.WriteLine("***************************************\n");
            Console.WriteLine($"Total milliseconds: {timewatch.Elapsed.TotalMilliseconds:F4}"); // Total milliseconds: 108.8356
            Console.WriteLine($"Total seconds: {timewatch.Elapsed.TotalSeconds:F4}"); // Total seconds: 0.1088
            Console.WriteLine($"Total minutes: {timewatch.Elapsed.TotalMinutes:F4}"); // Total minutes: 0.0018
        }

    }
}

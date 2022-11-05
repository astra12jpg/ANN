using System;
using System.Collections.Generic;
using System.Collections;
using System.Text;
using System.Xml.Linq;

namespace Artificial_Network
{
    class Network
    {
        public ArrayList extend_backprop(int[] config, ArrayList net_input, ArrayList net_enter, int Epochs, double alpha, double error_limit, double HIGH, double LOW)
        {

            double[][,] weights = (double[][,])net_enter[0];
            double[][,] gradients = (double[][,])net_enter[1];
            double[][] bias = (double[][])net_enter[2];
            double[][] outputs = (double[][])net_enter[3];
            double[][] errors = (double[][])net_enter[4];

            double[][] xdata = (double[][])net_input[0]; double[][] ydata = (double[][])net_input[1];
            double[][] xtest = (double[][])net_input[2]; double[][] ytest = (double[][])net_input[3];
            double max = (double)net_input[4]; double min = (double)net_input[5]; int tr = (int)net_input[8];


            int index = 0; double[] rmse = new double[Epochs]; int count = 0;
            double[] cross = new double[Epochs]; double[] cross_data = new double[Epochs]; cross[0] = 100; cross_data[0] = 100;
            double[] input = new double[xdata[0].Length]; double[] target = new double[ydata[0].Length]; double bt = 0;
            double[][] outs = new double[config.GetLength(0) + 1][];


            double[][,] weights_temp = new double[config.Length + 1][,]; double[][] bias_temp = new double[config.Length + 1][];
            ArrayList sk = new ArrayList(); int step = 0; double[][] yfin_test = new double[xtest.Length][]; double[][] yfin_data = new double[xdata.Length][];
            while (index < Epochs)
            {
                for (int i = 0; i < xdata.Length; i++)
                {
                    input = xdata[i]; target = ydata[i];
                    outputs = forward(weights, bias, input, config);
                    sk = backward(weights, gradients, outputs, bias, errors, input, target, config, alpha);
                    weights = (double[][,])sk[0]; bias = (double[][])sk[3];
                }
                double test = 0;
                for (int kj = 0; kj < xtest.Length; kj++)
                {
                    outputs = forward(weights, bias, xtest[kj], config); yfin_test[kj] = outputs[config.Length];
                    test = test + loss_softmax(outputs, ytest[kj], config);
                }
                cross[index] = test / xtest.Length;

                double test_data = 0;
                for (int j = 0; j < xdata.Length; j++)
                {
                    outputs = forward(weights, bias, xdata[j], config); yfin_data[j] = outputs[config.Length];
                    test_data = test_data + loss_softmax(outputs, ydata[j], config);
                }
                cross_data[index] = test_data / xdata.Length;
                if (index > 0)
                {
                    if (cross[index] - cross[index - 1] < error_limit)
                    {
                        double nc_test = test_softmax(yfin_test, ytest); double nc_data = test_softmax(yfin_data, ydata);
                        if (nc_data > 95 && nc_test > 95)
                        {
                            break;
                        }
                        else
                        {
                            Random rnd = new Random();
                            for (int kp = 0; kp < weights.Length; kp++)
                            {
                                for (int kt = 0; kt < weights[kp].GetLength(0); kt++)
                                {
                                    for (int kr = 0; kr < weights[kp].GetLength(1); kr++)
                                    {
                                        weights[kp][kt, kr] = weights[kp][kt, kr] + 0.1 * rnd.NextDouble();
                                    }
                                }
                            }
                            for (int km = 0; km < bias.Length; km++)
                            {
                                for (int kn = 0; kn < bias[km].Length; kn++)
                                {
                                    bias[km][kn] = bias[km][kn] + 0.1 * rnd.NextDouble();
                                }
                            }
                        }
                    }
                }
                index++;
            }

            ArrayList results = new ArrayList();

            results.Add(weights); results.Add(gradients); results.Add(bias);
            results.Add(cross); results.Add(index); results.Add(cross_data);

            return results;

        }
        public ArrayList network_init(int[] config, int input_size, int output_size)
        {
            ArrayList start = new ArrayList();

            double[][,] weights = new double[config.GetLength(0) + 1][,]; Random rnd = new Random();
            double[][,] grad = new double[config.GetLength(0) + 1][,];


            for (int i = 0; i < config.GetLength(0); i++)
            {
                if (i == 0)
                {
                    double[,] tempw = new double[config[0], input_size];
                    double[,] tempgr = new double[config[0], input_size];

                    for (int n = 0; n < config[0]; n++)
                    {


                        for (int m = 0; m < input_size; m++)
                        {

                            tempw[n, m] = rnd.NextDouble(); tempgr[n, m] = 0;
                        }

                    }
                    weights[i] = tempw; grad[i] = tempgr;
                }
                else
                {
                    double[,] tempw = new double[config[i], config[i - 1]];
                    double[,] tempgr = new double[config[i], config[i - 1]];

                    for (int n = 0; n < config[i]; n++)
                    {
                        for (int m = 0; m < config[i - 1]; m++)
                        {
                            tempw[n, m] = rnd.NextDouble(); tempgr[n, m] = 0;
                        }

                    }
                    weights[i] = tempw; grad[i] = tempgr;
                }
            }
            double[,] tempw1 = new double[output_size, config[config.GetLength(0) - 1]];
            double[,] tempgr1 = new double[output_size, config[config.GetLength(0) - 1]];

            for (int n = 0; n < output_size; n++)
            {
                for (int m = 0; m < config[config.GetLength(0) - 1]; m++)
                {
                    tempw1[n, m] = rnd.NextDouble(); tempgr1[n, m] = 0;

                }
            }

            weights[config.GetLength(0)] = tempw1; grad[config.GetLength(0)] = tempgr1;

            start.Add(weights); start.Add(grad);

            double[][] bias = new double[config.GetLength(0) + 1][];
            double[][] outputs = new double[config.GetLength(0) + 1][];
            double[][] errors = new double[config.GetLength(0) + 1][];

            for (int i = 0; i < config.GetLength(0); i++)
            {
                double[] tempou = new double[config[i]]; double[] tempb = new double[config[i]];
                double[] temper = new double[config[i]];

                for (int j = 0; j < config[i]; j++)
                {
                    tempou[j] = 0; tempb[j] = 0; temper[j] = 0;

                }

                outputs[i] = tempou; bias[i] = tempb; errors[i] = temper;
            }
            double[] tempou1 = new double[output_size]; double[] tempb1 = new double[output_size];
            double[] temper1 = new double[output_size];

            for (int j = 0; j < output_size; j++)
            {
                tempou1[j] = 0; tempb1[j] = 0; temper1[j] = 0;
            }
            outputs[config.GetLength(0)] = tempou1; bias[config.GetLength(0)] = tempb1; errors[config.GetLength(0)] = temper1;

            start.Add(bias); start.Add(outputs); start.Add(errors);

            return start;
        }

        public double[][] forward(double[][,] weights, double[][] bias, double[] input, int[] config)
        {
            double[][] outs = new double[config.Length + 1][];

            for (int i = 0; i <= config.Length; i++)
            {
                if (i == 0)
                {
                    outs[i] = multipl21D(weights[i], input);
                    outs[i] = add1D(outs[i], bias[i]);
                    outs[i] = transfer(outs[i]);
                }
                else
                {
                    outs[i] = multipl21D(weights[i], outs[i - 1]);
                    outs[i] = add1D(outs[i], bias[i]);
                    if (i == config.Length) { outs[i] = softmax_inline(outs[i]); } // Za cross-enropy pristup!
                    else { outs[i] = transfer(outs[i]); }
                }

            }
            return outs;
        }
        public double[] simulate(double[][,] weights, double[][] bias, double[] input, int[] config, int num_class, int high, int low, double maxx, double minix)
        {
            double[][] outs = new double[config.Length + 1][];
            double[] temp = new double[num_class];

            for (int k = 0; k < input.Length; k++)
            {
                input[k] = (high - low) * (input[k] - minix) / (maxx - minix) + low;
            }

            for (int i = 0; i <= config.Length; i++)
            {
                if (i == 0)
                {
                    outs[i] = multipl21D(weights[i], input);
                    outs[i] = add1D(outs[i], bias[i]);
                    outs[i] = transfer(outs[i]);
                }
                else
                {
                    outs[i] = multipl21D(weights[i], outs[i - 1]);
                    outs[i] = add1D(outs[i], bias[i]);
                    if (i == config.Length) { outs[i] = softmax_inline(outs[i]); temp = outs[i]; } // Za cross-enropy pristup!
                    else { outs[i] = transfer(outs[i]); }

                }

            }
            double maks = 0;
            for (int j = 0; j < temp.Length; j++)
            {
                if (temp[j] > maks) { maks = temp[j]; }
            }
            for (int j = 0; j < temp.Length; j++)
            {
                if (temp[j] == maks) { temp[j] = 1; }
                else { temp[j] = 0; }
            }
            return temp;
        }

        public ArrayList backward(double[][,] weights, double[][,] gradients, double[][] outputs, double[][] bias, double[][] errors, double[] input, double[] target, int[] config, double alpha)
        {
            ArrayList track = new ArrayList();

            for (int i = config.Length; i >= 0; i--)
            {
                if (i == config.Length)
                {
                    double[] temp = deriv1D(outputs[i]);
                    errors[i] = minus1D(outputs[i], target);
                    //errors[i] = dotprod1D(temp, errors[i]); // Za cross - entropy ovaj deo ISKLJUCITI! 
                    gradients[i] = error_weigths(errors[i], outputs[i - 1]);
                }
                else if (i == 0)
                {
                    double[] temp = deriv1D(outputs[i]);
                    errors[i] = multipl21D(transp(weights[i + 1]), errors[i + 1]);
                    errors[i] = dotprod1D(temp, errors[i]);
                    gradients[i] = error_weigths(errors[i], input);
                }
                else
                {
                    double[] temp = deriv1D(outputs[i]);
                    errors[i] = multipl21D(transp(weights[i + 1]), errors[i + 1]);
                    errors[i] = dotprod1D(temp, errors[i]);
                    gradients[i] = error_weigths(errors[i], outputs[i - 1]);
                }

            }
            //CORRECTIONS!
            for (int j = 0; j <= config.Length; j++)
            {
                weights[j] = minus(weights[j], const2D(gradients[j], alpha));
                bias[j] = minus1D(bias[j], const1D(errors[j], alpha));
            }
            track.Add(weights); track.Add(gradients); track.Add(outputs);
            track.Add(bias); track.Add(errors);

            return track;

        }

        public double[,] error_weigths(double[] a, double[] b)
        {
            int dim1 = a.Length; int dim2 = b.Length;
            double[,] tc = new double[dim1, dim2];

            for (int i = 0; i < dim1; i++)
            {
                for (int j = 0; j < dim2; j++)
                {
                    tc[i, j] = a[i] * b[j];
                }
            }
            return tc;
        }
        public double[] transfer(double[] a)
        {
            double[] temp = new double[a.Length];

            for (int i = 0; i < a.Length; i++)
            {
                temp[i] = 1 / (1 + Math.Pow(2.72, -a[i])); // logsig
            }

            return temp;
        }

        public double[] deriv1D(double[] a)
        {
            double[] temp = new double[a.Length];

            for (int i = 0; i < a.Length; i++)
            {
                temp[i] = a[i] * (1 - a[i]);
            }

            return temp;
        }

        public double dist(double[] a, double[] b)
        {
            double sum = 0;
            for (int i = 0; i < a.Length; i++)
            {
                sum = sum + Math.Pow(a[i] - b[i], 2);
            }

            return (double)sum / a.Length;
        }
        public double[] softmax_inline(double[] a)
        {
            int dim1 = a.Length; double sumt = 0;

            for (int i = 0; i < dim1; i++)
            {
                sumt = sumt + Math.Exp(a[i]);
            }
            for (int j = 0; j < dim1; j++)
            {
                a[j] = Math.Exp(a[j]) / sumt;
            }
            return a;
        }
        public double loss_softmax(double[][] a, double[] b, int[] config)
        {
            int dim1 = a.Length; double tu = 0;

            for (int j = 0; j < a[config.Length].GetLength(0); j++)
            {
                if (a[config.Length][j] > 0) { tu = tu + b[j] * Math.Log(a[config.Length][j]); }
            }
            tu = -tu / dim1;
            return tu;
        }
        public double test_softmax(double[][] a, double[][] b)
        {
            int dim1 = a.Length; double tp = 0;
            double mintemp = 1000; int count; double stats = 0; double[] temp = new double[a[1].Length];

            for (int i = 0; i < dim1; i++)
            {
                count = 0;
                for (int j = 0; j < a[i].Length; j++)
                {
                    temp[j] = 1 - a[i][j];
                    if (temp[j] < mintemp) { mintemp = temp[j]; count = j; }
                }
                if (b[i][count] == 1) stats++;
                mintemp = 1000;
            }
            return 100 * stats / dim1;
        }

        public double[][] import_file(string path)
        {
            Microsoft.Office.Interop.Excel.Workbook xlWorkBook;
            Microsoft.Office.Interop.Excel.Worksheet xlWorkSheet;
            Microsoft.Office.Interop.Excel.Range range;
            Microsoft.Office.Interop.Excel.Application xlApp;

            int row = 0;
            int col = 0;

            xlApp = new Microsoft.Office.Interop.Excel.Application();
            xlWorkBook = xlApp.Workbooks.Open(path, 0, true, 5, "", "", true, Microsoft.Office.Interop.Excel.XlPlatform.xlWindows, "\t", false, false, 0, true, 1, 0);
            xlWorkSheet = (Microsoft.Office.Interop.Excel.Worksheet)xlWorkBook.Worksheets.get_Item(1);

            range = xlWorkSheet.UsedRange;
            row = range.Rows.Count;
            col = range.Columns.Count;
            double[,] temp = new double[row, col];
            double[][] outs = new double[row][];

            for (int rCnt = 1; rCnt <= row; rCnt++)
            {
                for (int cCnt = 1; cCnt <= col; cCnt++)
                {
                    temp[rCnt - 1, cCnt - 1] = (range.Cells[rCnt, cCnt] as Microsoft.Office.Interop.Excel.Range).Value2;
                }
            }

            for (int i = 0; i < row; i++)
            {
                double[] temp1 = new double[temp.GetLength(1)];
                for (int j = 0; j < col; j++)
                {
                    temp1[j] = temp[i, j];
                }
                outs[i] = temp1;
            }

            xlWorkBook.Close(true, null, null);
            xlApp.Quit();

            return outs;

        }
        public ArrayList split_test(double[][] Data, int num_output, double ratio, int student)
        {
            double test = ratio * Data.GetLength(0); int tar = Data[0].Length;
            int dim1 = Convert.ToInt32(test); int dim2 = Data.GetLength(0) - dim1;
            double[][] xdata = new double[dim1][]; double[][] xtest = new double[dim2][];
            double[][] ydata = new double[dim1][]; double[][] ytest = new double[dim2][];

            ArrayList st = new ArrayList();

            Random rnd = new Random(0);
            for (int i = 0; i < Data.Length; ++i)
            {
                int r = rnd.Next(i, Data.Length);
                double[] tmp = Data[r];
                Data[r] = Data[i];
                Data[i] = tmp;
            }
            for (int i = 0; i < dim1; ++i)
            {
                double[] temp1 = new double[tar - num_output]; double[] temp2 = new double[num_output];

                for (int m = 0; m < tar - num_output; m++)
                {
                    temp1[m] = Data[i][m];

                }
                for (int n = 0; n < num_output; n++)
                {
                    temp2[n] = Data[i][n + tar - num_output];
                }
                xdata[i] = temp1;
                ydata[i] = temp2;

            }
            for (int i = 0; i < dim2; ++i)
            {
                double[] temp1 = new double[tar - num_output]; double[] temp2 = new double[num_output];

                for (int m = 0; m < tar - num_output; m++)
                {
                    temp1[m] = Data[i + dim1][m];
                }
                for (int n = 0; n < num_output; n++)
                {
                    temp2[n] = Data[i + dim1][n + tar - num_output];
                }
                xtest[i] = temp1;
                ytest[i] = temp2;
            }
            if (student == 1) { xdata = studentize2D(xdata); xtest = studentize2D(xtest); }
            st.Add(xdata); st.Add(ydata);
            st.Add(xtest); st.Add(ytest);
            return st;
        }

        public double[][] studentize2D(double[][] a)
        {
            int dim1 = a.Length; int dim2 = a[0].Length;

            for (int j = 0; j < dim2; j++)
            {
                double average = 0;
                for (int i = 0; i < dim1; i++)
                {
                    average = average + a[i][j];
                }
                average = average / dim1; double sum = 0;
                for (int i = 0; i < dim1; i++)
                {
                    sum = sum + Math.Pow(a[i][j] - average, 2);
                }
                double st_dev = sum / (dim1 - 1);
                for (int i = 0; i < dim1; i++)
                {
                    a[i][j] = (a[i][j] - average) / st_dev;
                }
            }
            return a;
        }
        public double[] studentize1D(double[] a)
        {
            int dim1 = a.Length;

            double average = 0;
            for (int i = 0; i < dim1; i++)
            {
                average = average + a[i];
            }
            average = average / dim1; double sum = 0;
            for (int i = 0; i < dim1; i++)
            {
                sum = sum + Math.Pow(a[i] - average, 2);
            }
            double st_dev = sum / (dim1 - 1);
            for (int i = 0; i < dim1; i++)
            {
                a[i] = (a[i] - average) / st_dev;
            }
            return a;
        }
        public ArrayList data_proces(ArrayList su, int num_input, int num_output, double HIGH, double LOW)
        {
            ArrayList st = new ArrayList();

            double[][] xdata = (double[][])su[0]; double[][] ydata = (double[][])su[1];
            double[][] xtest = (double[][])su[2]; double[][] ytest = (double[][])su[3];

            double maxiy = Math.Max(ydata[0][0], ydata[0][num_output - 1]); double miniy = Math.Min(ydata[0][0], ydata[0][num_output - 1]);
            double maxix = Math.Max(xdata[0][0], xdata[0][num_input - 1]); double minix = Math.Min(xdata[0][0], xdata[0][num_input - 1]);

            int dim1 = xdata.GetLength(0); int dim2 = xtest.GetLength(0);

            for (int i = 0; i < dim1; i++)
            {
                double testmaxy = Math.Max(ydata[i][0], ydata[i][num_output - 1]); double testminiy = Math.Min(ydata[i][0], ydata[i][num_output - 1]);
                if (testmaxy > maxiy) { maxiy = testmaxy; }
                if (testminiy < miniy) { miniy = testminiy; }
                double testmax = Math.Max(xdata[i][0], xdata[i][num_input - 1]); double testminix = Math.Min(xdata[i][0], xdata[i][num_input - 1]);
                if (testmax > maxix) { maxix = testmax; }
                if (testminix < minix) { minix = testminix; }

            }

            for (int i = 0; i < dim2; i++)
            {
                double testmaxys = Math.Max(ytest[i][0], ytest[i][num_output - 1]); double testminiys = Math.Min(ytest[i][0], ytest[i][num_output - 1]);
                if (testmaxys > maxiy) { maxiy = testmaxys; }
                if (testminiys < miniy) { miniy = testminiys; }
                double testmaxs = Math.Max(xtest[i][0], xtest[i][num_input - 1]); double testminixs = Math.Min(xtest[i][0], xtest[i][num_input - 1]);
                if (testmaxs > maxix) { maxix = testmaxs; }
                if (testminixs < minix) { minix = testminixs; }

            }

            for (int i = 0; i < dim1; i++)
            {
                for (int j = 0; j < num_input; j++)
                {
                    xdata[i][j] = (HIGH - LOW) * (xdata[i][j] - minix) / (maxix - minix) + LOW;
                }
            }

            for (int i = 0; i < dim2; i++)
            {
                for (int j = 0; j < num_input; j++)
                {
                    xtest[i][j] = (HIGH - LOW) * (xtest[i][j] - minix) / (maxix - minix) + LOW;
                }
            }
            int tr = 0;
            if (maxiy > 1)
            {
                tr = 1;
            }
            st.Add(xdata); st.Add(ydata); st.Add(xtest); st.Add(ytest);
            st.Add(maxiy); st.Add(miniy); st.Add(maxix); st.Add(minix);
            st.Add(tr);

            return st;
        }

        public void stats(ArrayList st, ArrayList results, int[] config)
        {
            double[][,] weights_final = (double[][,])results[0]; double[][] bias_final = (double[][])results[2];

            double[][] ytest = (double[][])st[3]; double[][] temp = new double[config.Length + 1][];
            double[][] xtest = (double[][])st[2]; double[][] yfinal = new double[ytest.Length][]; double sums; double sums_data;

            double[][] xdata = (double[][])st[0]; double[][] ydata = (double[][])st[1]; double[][] yfinal_data = new double[ydata.Length][];
            double[][] temp_data = new double[config.Length + 1][];

            for (int i = 0; i < ytest.Length; i++)
            {
                temp = forward(weights_final, bias_final, xtest[i], config);
                yfinal[i] = temp[config.Length];
            }

            Console.WriteLine("Number of epochs:{0}", (int)results[4]);

            double nc = test_softmax(yfinal, ytest);

            for (int i = 0; i < ydata.Length; i++)
            {
                temp_data = forward(weights_final, bias_final, xdata[i], config);
                yfinal_data[i] = temp_data[config.Length];
            }

            double nc1 = test_softmax(yfinal_data, ydata); double nc2 = (nc * xdata.Length + nc1 * xtest.Length) / (xdata.Length + xtest.Length);

            Console.WriteLine("Percentage of reproduction on test set:{0}%", Math.Round(nc, 2));
            Console.WriteLine("Percentage of reproduction on training set:{0}%", Math.Round(nc1, 2));
            Console.WriteLine("Percentage of reproduction on whole set:{0}%", Math.Round(nc2, 2));
        }

        public double[,] add(double[,] a, double[,] b)
        {
            int dim1 = a.GetLength(0); int dim2 = a.GetLength(1);

            double[,] c = new double[dim1, dim2];
            for (int i = 0; i < dim1; i++)
            {
                for (int j = 0; j < dim2; j++)
                {
                    c[i, j] = a[i, j] + b[i, j];
                }
            }

            return c;
        }

        public double[] add1D(double[] a, double[] b)
        {
            int dim1 = a.GetLength(0);

            double[] c = new double[dim1];

            for (int i = 0; i < dim1; i++)
            {
                c[i] = a[i] + b[i];

            }

            return c;
        }


        public double[,] minus(double[,] a, double[,] b)
        {
            int dim1 = a.GetLength(0); int dim2 = a.GetLength(1);

            double[,] c = new double[dim1, dim2];
            for (int i = 0; i < dim1; i++)
            {
                for (int j = 0; j < dim2; j++)
                {
                    c[i, j] = a[i, j] - b[i, j];
                }
            }

            return c;
        }

        public double[] minus1D(double[] a, double[] b)
        {
            int dim1 = a.GetLength(0);

            double[] c = new double[dim1];

            for (int i = 0; i < dim1; i++)
            {
                c[i] = a[i] - b[i];
            }

            return c;
        }

        public double[,] const2D(double[,] a, double s)
        {
            int dim1 = a.GetLength(0); int dim2 = a.GetLength(1);

            double[,] c = new double[dim1, dim2];

            for (int i = 0; i < dim1; i++)
            {
                for (int j = 0; j < dim2; j++)
                {
                    c[i, j] = s * a[i, j];
                }
            }

            return c;

        }

        public double[] const1D(double[] a, double s)
        {
            int dim = a.Length;

            double[] c = new double[dim];

            for (int i = 0; i < dim; i++)
            {
                c[i] = s * a[i];
            }

            return c;

        }
        public double[,] dotprod(double[,] a, double[,] b)
        {
            int dim1 = a.GetLength(0); int dim2 = a.GetLength(1);

            double[,] c = new double[dim1, dim2];
            for (int i = 0; i < dim1; i++)
            {
                for (int j = 0; j < dim2; j++)
                {
                    c[i, j] = a[i, j] * b[i, j];
                }
            }

            return c;
        }

        public double[] dotprod1D(double[] a, double[] b)
        {
            int dim1 = a.Length;

            double[] c = new double[dim1];
            for (int i = 0; i < dim1; i++)
            {

                c[i] = a[i] * b[i];

            }

            return c;
        }


        public double[,] multipl(double[,] a, double[,] b)
        {
            int dim1 = a.GetLength(0); int dim2 = a.GetLength(1); int dim3 = b.GetLength(1);

            double[,] c = new double[dim1, dim3];
            for (int i = 0; i < dim1; i++)
            {
                for (int j = 0; j < dim3; j++)
                {
                    double sum = 0;
                    for (int k = 0; k < dim2; k++)
                    {
                        sum = sum + a[i, k] * b[k, j];
                    }
                    c[i, j] = sum;
                }

            }

            return c;
        }

        public double[] multipl21D(double[,] a, double[] b)
        {
            int dim1 = a.GetLength(0); int dim2 = b.GetLength(0);

            double[] c = new double[dim1];

            for (int i = 0; i < dim1; i++)
            {
                double sum = 0;
                for (int j = 0; j < dim2; j++)
                {
                    sum = sum + a[i, j] * b[j];

                }
                c[i] = sum;

            }

            return c;
        }

        public double[,] transp(double[,] a)
        {
            int dim1 = a.GetLength(0); int dim2 = a.GetLength(1);

            double[,] c = new double[dim2, dim1];

            for (int i = 0; i < dim1; i++)
            {
                for (int j = 0; j < dim2; j++)
                {
                    c[j, i] = a[i, j];
                }
            }

            return c;
        }

        public double[,] mergeMat_column(double[,] a, double[,] b)
        {
            int dim1 = a.GetLength(0); int dim2 = b.GetLength(0);
            int dim3 = a.GetLength(1); int dim4 = b.GetLength(1);

            double[,] c = new double[dim1, dim3 + dim4];
            for (int i = 0; i < dim1; i++)
            {
                for (int j = 0; j < dim3; j++)
                {
                    c[i, j] = a[i, j];
                }
            }

            for (int i = 0; i < dim1; i++)
            {
                for (int j = 0; j < dim4; j++)
                {
                    c[i, j + dim3] = b[i, j];
                }
            }

            return c;
        }


    }
}


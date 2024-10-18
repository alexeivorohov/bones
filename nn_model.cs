using System.Diagnostics;

public class Adam
{
    double alpha;
    double beta;
    double eps;
    public Adam(double alpha, double beta, double eps)
    {

    }
}

// Класс моделей
class NN_Model 
{
    Sequential SEQ;
    LossF loss;
    Dictionary<int, string> columns;
 
    public double best_score;
    public Sequential best_model;
    
    public NN_Model(Sequential SEQ, LossF loss)
    {
        this.SEQ = SEQ;
        this.loss = loss;
        best_model = SEQ;
    }

    public void SaveState(string src)
    {
        FileStream fs = new(src, FileMode.OpenOrCreate);
        StreamWriter str = new(fs);

        str.WriteLine(SEQ.info_str()+"\n\n");

        for(int n=0; n<SEQ.Nlayers; n++)
            str.WriteLine(SEQ[n].info_str()+"\n\n");

        str.Close();
        fs.Close();

        Console.WriteLine("Model saved at: " + src);        
    }
    public void LoadState(string src)
    {
        FileStream fs = new(src, FileMode.Open);
        StreamReader str = new(fs);

        
        Sequential SEQ_ = new(str);

        str.Close();
        fs.Close();

        SEQ = SEQ_;

        Console.WriteLine($"Loaded model from {src}");

        Summary();
    }

    public double[] fit(DataFrame X, DataFrame Y, int epochs, double lr, Score scorer)
    {
        int smplN = Y.shape[0];
        int conv_timer = 0;        

        columns = Y.Columns;

        double[] error = new double[epochs];
        SEQ.LR = lr;

        bool eval_swtch = (scorer != null);
        if (eval_swtch) { best_score = 0; }
        else { best_score = 3 * loss.eval(SEQ.forward(X[0]), X[0]); }


        for (int eph = 0; eph < epochs; eph++)
        {
            SEQ.train_mode = true;
            double ev1 = 0;

            double eph_err = 0;
            for (int smpl = 0; smpl < smplN; smpl++)
            {
                var outp = SEQ.forward(X[smpl]);

                eph_err += loss.eval(outp, Y[smpl]);

                SEQ.backward(X[smpl], loss.grad(outp, Y[smpl]));
            }

            SEQ.train_mode = false;
            if (eval_swtch) { ev1 = scorer.Eval(predict(X), Y)[^1]; } 


            eph_err /= smplN;
            error[eph] = eph_err;
            

            var msg = $"Epoch {eph}, loss: {eph_err:f6}";

            if (eval_swtch)
            {
                msg += $" score: {ev1:f4}";
                if (ev1 > best_score)
                {
                    best_model = SEQ;
                    best_score = ev1;
                    Console.WriteLine($"Best score {best_score:f5}");
                }

                if (best_score - ev1 < 0.1 ) { conv_timer += 1; }
                else { conv_timer = 0; }

                if (conv_timer > 70)
                {
                    Console.WriteLine("Convergence");
                    break;
                }
            }
            else
            {
                if (eph_err < best_score)
                {
                    best_model = SEQ;
                    best_score = eph_err;
                }
            }

            Console.WriteLine(msg);
        }

        SEQ = best_model;

        Console.WriteLine();

        return error;
    }

    public DataFrame predict(DataFrame X)
    {
        SEQ.train_mode = false;

        string[] new_cols = new string[columns.Count];
        int n = 0;
        foreach (var clmn in columns)
        {
            new_cols[n] = clmn.Value + "_p";
            n += 1;
        }

        FileStream fs = new FileStream("pred_mem.csv", FileMode.Create);
        StreamWriter str = new StreamWriter(fs);

        foreach(var row in X.data)
        {
            var tmp = $"{row.Key}";

            double[] y_p = SEQ.forward(row.Value);

            foreach (var yi in y_p)
                tmp += $" {yi:f8}";

            str.WriteLine(tmp);
            //Console.WriteLine($"{row.Key}: {y_p[0]:f8}");
        }
        str.Close();
        fs.Close();

        DataFrame outp = new(new_cols);
        outp.ReadCSV("pred_mem.csv", true, " ");

        return outp; // new DataFrame(outp, new_cols);
    }

    public double[] predict(double[] X)
    {
        //SEQ.train_mode = false;

        return SEQ.forward(X);
    }

    public void Summary()
    {
        Console.WriteLine("NN info");
        Console.WriteLine("Module\tShape");
        Console.WriteLine("--------------------------");
        for (int n = 0; n < SEQ.Nlayers; n++)
        {
            string outp = SEQ[n].name;

            outp += $"\t[{SEQ[n].shape[0]}, {SEQ[n].shape[1]}]\t LR:{SEQ[n].LR:f6}";

            Console.WriteLine(outp);
        }
         
        Console.WriteLine();

    }

    public void DebugForward(double[] X)
    {
        string msg = "X";
        Console.WriteLine("Test input:");
        foreach (var x_ in X)
            msg += $"\t{x_:f3}";
        Console.WriteLine(msg);

        predict(X);

        
        for (int n = 0; n < SEQ.Nlayers; n++)
        {
            msg = $"{n}: {SEQ[n].name}|";

            foreach (var x_ in SEQ[n].outp)
                msg += $"\t{x_:f3}";

            Console.WriteLine(msg);
        }

        Console.WriteLine();
    }

    public void DebugBackward(double[] X, double[] Y)
    {
        SEQ.train_mode = true;
        var outp = predict(X);
        
        Console.WriteLine("Test input:");

        string log = "Y_p:";
        foreach (var x_ in outp)
            log += $"\t{x_:f3}";

        log += "\nY:";
        foreach (var y_ in Y)
            log += $"\t{y_:f3}";
        Console.WriteLine(log);

        for (int n = SEQ.Nlayers-1; n >=0; n--)
        {
            var dY = loss.grad(outp, Y);
            SEQ.backward(X, dY);

            log = $"{n}: {SEQ[n].name}|";

            foreach (var x_ in SEQ[n].grad)
                log += $"\t{x_:f3}";

            Console.WriteLine(log);
        }

        SEQ.train_mode = false;

        Console.WriteLine();
    }

    public void eval_mode()
    {
        SEQ.train_mode = false;
    }

    public void train_mode()
    {
        SEQ.train_mode = true;
    }
}

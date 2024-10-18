using System.Diagnostics;

// Нормировки
public class StdScaler
{
    double[] means;
    double[] std;
    string[] columns;
    int L;

    public StdScaler(DataFrame DF)
    {
        L = DF.shape[1];
        columns = new string[L];
        std = new double[L];
        means = new double[L];

        foreach (var col in DF.Columns)
            columns[col.Key] = col.Value;

        Fit(DF);
    }
    public void Fit(DataFrame DF)
    {
        double[] m2 = new double[L];

        foreach (var row in DF.data)
        {
            for (int i = 0; i < L; i++)
            {
                means[i] += row.Value[i];
                m2[i] += Math.Pow(row.Value[i], 2);
            }
        }

        for (int i = 0; i < L; i++)
        {
            means[i] /= DF.shape[0];
            m2[i] /= DF.shape[0];
            std[i] = Math.Sqrt(m2[i] - (means[i] * means[i]));
        }
    }

    public DataFrame Transform(DataFrame DF)
    {
        Dictionary<int, double[]> tmp = new();

        foreach (var row in DF.data)
        {
            tmp[row.Key] = new double[L];
            for (int i = 0; i < L; i++)
                tmp[row.Key][i] = (row.Value[i] - means[i]) / 3 / std[i];
        }

        DataFrame outp = new(tmp, columns);
        return outp;
    }
    public DataFrame InverseTransform(DataFrame DF)
    {
        DataFrame outp = new(columns);
        double[] tmp;

        foreach (var row in DF.data)
        {
            tmp = new double[L];

            for (int i = 0; i < L; i++)
                tmp[i] = row.Value[i] * 3 * std[i] + means[i];

            outp.data.Add(row.Key, tmp);
        }

        return outp;
    }

    public void InverseTransform(DataFrame DF, int[] ID0, int[] ID1)
    {
        foreach (var row in DF.data)
        {
            for (int i = 0; i < ID0.Length; i++)

                DF.data[row.Key][ID0[i]] = row.Value[ID0[i]] * 3 * std[ID1[i]] + means[ID1[i]];

        }
    }

    public void Info()
    {
        string outp = "Chars:";
        foreach (var clmn in columns)
            outp += "\t" + clmn;
        Console.WriteLine(outp);
        Console.WriteLine("---------------------------");

        outp = "Mean";
        foreach (var s in means)
            outp += $"\t{s:f6}";
        Console.WriteLine(outp);

        outp = "Std";
        foreach (var s in std)
            outp += $"\t{s:f6}";
        Console.WriteLine(outp);

        Console.WriteLine();
    }
}

public class MaxScaler
{
    double[] mx;
    double[] mn;
    string[] columns;
    int L;
    bool WithMin;
    double eta;

    public MaxScaler(DataFrame DF, bool WithMin, double eta)
    {
        L = DF.shape[1];
        columns = new string[L];
        this.WithMin = WithMin;
        this.eta = eta;

        mx = new double[L];
        mn = new double[L];
        for (int i = 0; i < L; i++)
        {
            mx[i] = DF[0][i];
            mn[i] = DF[0][i];
        }

        foreach (var col in DF.Columns)
            columns[col.Key] = col.Value;

        Fit(DF);
    }
    public void Fit(DataFrame DF)
    {
        foreach (var row in DF.data)
        {
            for (int i = 0; i < L; i++)
            {
                if (mx[i] < row.Value[i]) mx[i] = row.Value[i];
                if (mn[i] > row.Value[i]) mn[i] = row.Value[i];
            }
        }
    }

    public DataFrame Transform(DataFrame DF)
    {
        Dictionary<int, double[]> tmp = new();

        if (WithMin)
        {
            foreach (var row in DF.data)
            {
                tmp[row.Key] = new double[L];
                for (int i = 0; i < L; i++)
                    tmp[row.Key][i] = row.Value[i] / mx[i] * eta;
            }
        }
        else
        {
            foreach (var row in DF.data)
            {
                tmp[row.Key] = new double[L];
                for (int i = 0; i < L; i++)
                    tmp[row.Key][i] = (row.Value[i] - mn[i]) / (mx[i] - mn[i]) * eta;
            }
        }
        DataFrame outp = new(tmp, columns);
        return outp;
    }
    public DataFrame InverseTransform(DataFrame DF)
    {
        Dictionary<int, double[]> tmp = new();

        if (WithMin)
        {
            foreach (var row in DF.data)
            {
                tmp[row.Key] = new double[L];
                for (int i = 0; i < L; i++)
                    tmp[row.Key][i] = row.Value[i] * mx[i] / eta;
            }
        }
        else
        {
            foreach (var row in DF.data)
            {
                tmp[row.Key] = new double[L];
                for (int i = 0; i < L; i++)
                    tmp[row.Key][i] = row.Value[i] * (mx[i] - mn[i]) / eta + mn[i];
            }
        }

        DataFrame outp = new(tmp, columns);
        return outp;
    }

    public void InverseTransform(DataFrame DF, int[] ID0, int[] ID1)
    {
        if (WithMin)
        {
            foreach (var row in DF.data)
            {
                for (int i = 0; i < L; i++)
                    DF.data[row.Key][ID0[i]] = row.Value[i] * mx[i] / eta;
            }
        }
        else
        {
            foreach (var row in DF.data)
            {

                for (int i = 0; i < L; i++)
                    DF.data[row.Key][ID0[i]] = row.Value[i] * (mx[i] - mn[i]) / eta + mn[i];
            }
        }
    }

    public void Info()
    {
        string outp = "Chars:";
        foreach (var clmn in columns)
            outp += "\t" + clmn;
        Console.WriteLine(outp);
        Console.WriteLine("---------------------------");

        outp = "Max";
        foreach (var s in mx)
            outp += $"\t{s:f6}";
        Console.WriteLine(outp);

        outp = "Min";
        foreach (var s in mn)
            outp += $"\t{s:f6}";
        Console.WriteLine(outp);

        Console.WriteLine();
    }
}

// Работа с данными
public class DataFrame
{
    public Dictionary<int, double[]> data;
    Dictionary<int, string> columns = new();

    public Dictionary<int, string> Columns { get { return columns; } }
    //public Dictionary<int, double[]> Data { get { return data; } set { data = value; } }

    int[] _shape = { 0, 0 };

    public int[] shape { get { return _shape; } }

    public DataFrame(string[] columns)
    {
        data = new();
        _shape[1] = columns.Length;
        for (int n = 0; n < _shape[1]; n++)
            this.columns.Add(n, columns[n]);
    }

    public DataFrame(Dictionary<int, double[]> D, string[] columns)
    {
        _shape[1] = columns.Length;
        for (int n = 0; n < _shape[1]; n++)
            this.columns.Add(n, columns[n]);

        data = new();
        foreach (var row in D)
        {
            Append(row.Value);
        }

        _shape[0] = data.Count;
    }

    public void Append(double[] row)
    {
        Debug.Assert(row.Length == _shape[1], string.Format("DF.shape[1]={0} and input of  len({1}) do not match", _shape[1], row.Length));

        if (data.TryAdd(shape[0], row))
            _shape[0] += 1;
    }

    public void ColumnwiseAdd(double[] add)
    {
        foreach (var key in data.Keys)
        {
            for (int i = 0; i < shape[1]; i++)
                data[key][i] += add[i];
        }
    }

    public void ColumnwiseMul(double[] mul)
    {
        foreach (var key in data.Keys)
        {
            for (int i = 0; i < shape[1]; i++)
                data[key][i] *= mul[i];
        }
    }

    public void ReadCSV(string src, bool with_ID, string? splitter)
    {
        if (splitter == null) { splitter = ","; }

        FileStream fs = new FileStream(src, FileMode.Open);
        StreamReader str = new StreamReader(fs);
        string[] line;


        while (str.EndOfStream == false)
        {
            line = str.ReadLine().Split(splitter);
            double[] inp;

            int row_len = line.Length;

            if (with_ID)
            {
                inp = new double[row_len - 1];
                for (int i = 0; i < row_len - 1; i++)
                    inp[i] = Convert.ToDouble(line[i + 1]);
                data.TryAdd(Convert.ToInt32(line[0]), inp);
            }
            else
            {
                inp = new double[row_len];
                for (int i = 0; i < row_len; i++)
                {
                    //Console.WriteLine(line[i]);
                    inp[i] = Convert.ToDouble(line[i]);
                }
                Append(inp);
            }

        }
        _shape[0] = data.Count;

        str.Close();
        fs.Close();
    }


    public DataFrame Concat(DataFrame DF, int axis)
    {
        DataFrame outp;
        if (axis == 0)
        {
            outp = this;
            foreach (var row in DF.data)
                outp.data.TryAdd(row.Key, row.Value);

            outp.shape[0] = outp.data.Count;

            return outp;
        }
        if (axis == 1)
        {
            string[] new_cols = new string[shape[1] + DF.shape[1]];
            for (int n = 0; n < shape[1]; n++)
                new_cols[n] = columns[n];
            for (int n = 0; n < DF.shape[1]; n++)
                new_cols[shape[1] + n] = DF.columns[n];

            outp = new(new_cols);

            double[] tmp;
            foreach (var k in data.Keys)
            {
                tmp = new double[shape[1] + DF.shape[1]];
                for (int n = 0; n < shape[1]; n++)
                    tmp[n] = data[k][n];
                for (int n = 0; n < DF.shape[1]; n++)
                    tmp[shape[1] + n] = DF.data[k][n];

                outp.data.TryAdd(k, tmp);
            }

            outp.shape[0] = outp.data.Count;
            return outp;
        }
        else
        {
            return null;
        }
    }

    public void Head(int num)
    {
        string outp = "ID";
        foreach (var clmn in columns)
            outp += "\t" + clmn.Value;
        Console.WriteLine(outp);
        Console.WriteLine("---------------------------");

        int idx = 0;
        foreach (var row in data)
        {
            if (idx < num)
            {
                outp = $"{row.Key}";
                for (int i = 0; i < row.Value.Length; i++)
                    outp += $"\t{row.Value[i]:f4}";

                Console.WriteLine(outp);
            }
            idx += 1;
            if (idx == num)
                break;
        }
        Console.WriteLine();
    }

    public void Print(params int[] keys)
    {

    }

    public void Info()
    {
        Console.WriteLine($"DataFrame Info: shape: [{shape[0]}, {shape[1]}]");

        Console.WriteLine();
    }

    public double[] pop(int ax, int id)
    {
        return null;
    }

    public DataFrame GetCols(int[] IDS)
    {
        int ID_L = IDS.Length;
        double[] tmp;

        string[] new_cols = new string[ID_L];
        for (int i = 0; i < ID_L; i++) { new_cols[i] = columns[IDS[i]]; }

        DataFrame outp = new(new_cols);

        foreach (var key in data.Keys)
        {
            tmp = new double[ID_L];
            for (int i = 0; i < ID_L; i++) { tmp[i] = data[key][IDS[i]]; }
            outp.data.Add(key, tmp);
        }
        outp.shape[0] = data.Count;

        return outp;
    }

    public double[] this[int idx]
    {
        get { return data[idx]; }
        set { data[idx] = value; }
    }

    public double[] this[Range[] rnges]
    {
        get { throw new NotImplementedException(); }
        set { throw new NotImplementedException(); }
    }

}
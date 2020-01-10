using MLProject1.CNN;
using Newtonsoft.Json;
using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;

namespace MLProject1
{
    public partial class ConfusionMatrixForm : Form
    {
        //0 = training, 1 = testing, 2 = validation
        int state = 0;
        string[] sets = new string[3] {"Training","Testing", "Validation"};
        EvaluationMetrics[] metrics = new EvaluationMetrics[3];

        public ConfusionMatrixForm()
        {
            InitializeComponent();

            metrics[0] = JsonConvert.DeserializeObject<EvaluationMetrics>(File.ReadAllText("TrainingPerformance.json"));
            metrics[1] = JsonConvert.DeserializeObject<EvaluationMetrics>(File.ReadAllText("TestingPerformance.json"));
            metrics[2] = JsonConvert.DeserializeObject<EvaluationMetrics>(File.ReadAllText("ValidationPerformance.json"));

            //metrics[0] = new EvaluationMetrics(new int[3, 3] { { 1, 2, 3 }, { 4, 5, 6 }, { 7, 8, 9 } }, 10);

            InitializeTable();
        }

        private void InitializeTable()
        {
            setLabel.Text = sets[state];

            int size = metrics[state].ConfusionMatrix.GetLength(0) + 1;
            view.Rows.Clear();
            view.RowTemplate.Height = 15;
            view.RowTemplate.MinimumHeight = 15;
            for (int i = 0; i < size; i++)
            {
                view.Rows.Add(new DataGridViewRow());
                view.Columns[i].Width = view.Size.Width / (size + 5);
                for(int j = 0; j < size; j++)
                {
                    if (j == 0)
                    {
                        view.Rows[i].Cells[j].Value = (char)('A' + i - 1);
                    }
                    else
                    {
                        if (i == 0)
                        {
                            view.Rows[i].Cells[j].Value = (char)('A' + j - 1);
                        }
                        else
                        {
                            double factor = (double)metrics[state].ConfusionMatrix[i - 1, j - 1] / metrics[state].Total;
                            view.Rows[i].Cells[j].Style.BackColor = Color.FromArgb(Math.Max(0, (int)(Math.Max(0, 255 - 255 * factor * 100))), 255, 255);
                        }
                    }
                }
            }
            
        }

        private void Button2_Click(object sender, EventArgs e)
        {
            state = (state + 1) % 3;

            InitializeTable();
        }

        private void Button1_Click(object sender, EventArgs e)
        {
            state = (state + 2) % 3;

            InitializeTable();
        }
    }
}

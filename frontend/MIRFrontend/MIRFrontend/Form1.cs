using System;
using System.Collections.Generic;
using System.Windows.Forms;
using Newtonsoft.Json;

namespace MIRFrontend
{
    public partial class Form1 : Form
    {
        private readonly ApiClient client = new ApiClient();

        public Form1()
        {
            InitializeComponent();
        }

        /// <summary>
        /// 폴더 선택 다이얼로그 출력
        /// </summary>
        private void btnSelectFolder_Click_1(object sender, EventArgs e)
        {
            using (FolderBrowserDialog dialog = new FolderBrowserDialog())
            {
                if (dialog.ShowDialog() == DialogResult.OK)
                {
                    txtSelectedFolder.Text = dialog.SelectedPath;
                }
            }
        }

        /// <summary>
        /// 검색 실행
        /// </summary>
        private async void btnSearch_Click(object sender, EventArgs e)
        {
            string folder = txtSelectedFolder.Text;
            string query = txtQuery.Text;

            if (string.IsNullOrWhiteSpace(folder))
            {
                MessageBox.Show("폴더를 선택하세요.");
                return;
            }

            try
            {
                lstResult.Items.Clear();

                // UI에서 선택한 확장자 수집
                var extensions = new List<string>();
                foreach (var item in checkedListBox1.CheckedItems)
                {
                    extensions.Add(item.ToString().TrimStart('.'));
                }

                string json = await client.SearchAsync(query, folder, extensions);

                var response = JsonConvert.DeserializeObject<SearchResponse>(json);

                if (response == null || response.Status != "success")
                {
                    MessageBox.Show("검색 실패 또는 응답 오류");
                    return;
                }

                foreach (var item in response.Results)
                {
                    lstResult.Items.Add($"{item.Score:F2} | {item.Path}");
                }
            }
            catch (Exception ex)
            {
                MessageBox.Show($"오류 발생: {ex.Message}");
            }
        }

        private TextBox txtSelectedFolder;

        private void InitializeComponent()
        {
            this.txtSelectedFolder = new TextBox();
            this.txtQuery = new TextBox();
            this.lstResult = new ListBox();
            this.btnSelectFolder = new Button();
            this.btnSearch = new Button();
            this.checkedListBox1 = new CheckedListBox();
            this.SuspendLayout();

            // txtSelectedFolder
            this.txtSelectedFolder.Location = new System.Drawing.Point(12, 43);
            this.txtSelectedFolder.Name = "txtSelectedFolder";
            this.txtSelectedFolder.Size = new System.Drawing.Size(421, 25);

            // txtQuery
            this.txtQuery.Location = new System.Drawing.Point(12, 93);
            this.txtQuery.Name = "txtQuery";
            this.txtQuery.Size = new System.Drawing.Size(421, 25);

            // lstResult
            this.lstResult.FormattingEnabled = true;
            this.lstResult.ItemHeight = 15;
            this.lstResult.Location = new System.Drawing.Point(12, 208);
            this.lstResult.Size = new System.Drawing.Size(529, 109);

            // btnSelectFolder
            this.btnSelectFolder.Location = new System.Drawing.Point(444, 43);
            this.btnSelectFolder.Name = "btnSelectFolder";
            this.btnSelectFolder.Size = new System.Drawing.Size(97, 23);
            this.btnSelectFolder.Text = "파일찾기";
            this.btnSelectFolder.UseVisualStyleBackColor = true;
            this.btnSelectFolder.Click += new EventHandler(this.btnSelectFolder_Click_1);

            // btnSearch
            this.btnSearch.Location = new System.Drawing.Point(444, 95);
            this.btnSearch.Name = "btnSearch";
            this.btnSearch.Size = new System.Drawing.Size(97, 23);
            this.btnSearch.Text = "검색";
            this.btnSearch.UseVisualStyleBackColor = true;
            this.btnSearch.Click += new EventHandler(this.btnSearch_Click);

            // checkedListBox1
            this.checkedListBox1.Items.AddRange(new object[] { ".txt", ".jpg" });
            this.checkedListBox1.Location = new System.Drawing.Point(444, 140);
            this.checkedListBox1.Size = new System.Drawing.Size(97, 44);

            // Form1
            this.ClientSize = new System.Drawing.Size(569, 338);
            this.Controls.Add(this.checkedListBox1);
            this.Controls.Add(this.btnSearch);
            this.Controls.Add(this.btnSelectFolder);
            this.Controls.Add(this.lstResult);
            this.Controls.Add(this.txtQuery);
            this.Controls.Add(this.txtSelectedFolder);
            this.Name = "Form1";
            this.ResumeLayout(false);
            this.PerformLayout();
        }

        private TextBox txtQuery;
        private ListBox lstResult;
        private Button btnSelectFolder;
        private Button btnSearch;
        private CheckedListBox checkedListBox1;

        private void checkedListBox1_SelectedIndexChanged(object sender, EventArgs e)
        {
        }
    }

    public class SearchResponse
    {
        public string Status { get; set; }
        public List<SearchResult> Results { get; set; }
    }

    public class SearchResult
    {
        public string Path { get; set; }
        public double Score { get; set; }
    }
}

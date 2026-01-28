using System;
using System.Collections.Generic;
using System.Net.Http;
using System.Threading.Tasks;

namespace MIRFrontend
{
    public sealed class ApiClient
    {
        private static readonly HttpClient Http = new HttpClient();
        //config/config.yaml에 있음 
        private const string BaseUrl = "http://127.0.0.1:8080";

        public async Task<string> SearchAsync(string query, string rootPath, List<string> extensions)
        {
            var url = $"{BaseUrl}/search?query={Uri.EscapeDataString(query)}&root_path={Uri.EscapeDataString(rootPath)}";

            if (extensions != null)
            {
                foreach (var ext in extensions)
                {
                    url += $"&extensions={Uri.EscapeDataString(ext)}";
                }
            }

            return await Http.GetStringAsync(url);
        }
    }
}

using System;
using System.Net.Http;
using System.Net.Http.Json;
using System.Threading.Tasks;
using CineFlowStudio.Desktop.Models;

namespace CineFlowStudio.Desktop.Services
{
    public interface IColabDiscoveryService
    {
        Task<ColabBridgeInfo?> DiscoverActiveColabAsync();
        Task<bool> VerifyConnectionAsync(string baseUrl);
    }

    public class ColabDiscoveryService : IColabDiscoveryService
    {
        private const string FirebaseRtdbEndpoint = "https://ai-video-studio-ecde9-default-rtdb.firebaseio.com/cineflow/bridge.json";
        private readonly HttpClient _httpClient;

        public ColabDiscoveryService(HttpClient? httpClient = null)
        {
            _httpClient = httpClient ?? new HttpClient { Timeout = TimeSpan.FromSeconds(6) };
            _httpClient.DefaultRequestHeaders.Add("User-Agent", "CineFlow-Desktop-Client");
        }

        public async Task<ColabBridgeInfo?> DiscoverActiveColabAsync()
        {
            try
            {
                var bridge = await _httpClient.GetFromJsonAsync<ColabBridgeInfo>(FirebaseRtdbEndpoint);
                if (bridge != null && !string.IsNullOrWhiteSpace(bridge.Url))
                {
                    // Clean trailing slash
                    bridge.Url = bridge.Url.TrimEnd('/');
                    return bridge;
                }
            }
            catch (Exception ex)
            {
                System.Diagnostics.Debug.WriteLine($"[AutoDiscovery] Firebase RTDB fetch warning: {ex.Message}");
            }

            return null;
        }

        public async Task<bool> VerifyConnectionAsync(string baseUrl)
        {
            if (string.IsNullOrWhiteSpace(baseUrl)) return false;

            try
            {
                string healthUrl = $"{baseUrl.TrimEnd('/')}/api/health";
                using var response = await _httpClient.GetAsync(healthUrl);
                return response.IsSuccessStatusCode;
            }
            catch
            {
                return false;
            }
        }
    }
}

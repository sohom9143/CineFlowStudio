using System;
using System.Collections.Generic;
using System.IO;
using System.Net.Http;
using System.Net.Http.Json;
using System.Text.Json;
using System.Threading.Tasks;
using CineFlowStudio.Desktop.Models;

namespace CineFlowStudio.Desktop.Services
{
    public interface IColabApiClient
    {
        void SetBaseUrl(string url);
        string GetBaseUrl();
        Task<List<CharacterModel>> GetCharactersAsync();
        Task<CharacterModel?> CreateCharacterAsync(string name, string description, string gender, string? imageFilePath = null);
        Task<GenerationResponse?> GenerateShotAsync(GenerationRequest request);
    }

    public class ColabApiClient : IColabApiClient
    {
        private string _baseUrl = "http://127.0.0.1:7860";
        private readonly HttpClient _httpClient;

        public ColabApiClient(HttpClient? httpClient = null)
        {
            _httpClient = httpClient ?? new HttpClient { Timeout = TimeSpan.FromMinutes(5) };
        }

        public void SetBaseUrl(string url)
        {
            _baseUrl = url.TrimEnd('/');
        }

        public string GetBaseUrl() => _baseUrl;

        public async Task<List<CharacterModel>> GetCharactersAsync()
        {
            try
            {
                var endpoint = $"{_baseUrl}/api/characters";
                var response = await _httpClient.GetAsync(endpoint);
                if (response.IsSuccessStatusCode)
                {
                    var list = await response.Content.ReadFromJsonAsync<List<CharacterModel>>();
                    return list ?? new List<CharacterModel>();
                }
            }
            catch (Exception ex)
            {
                System.Diagnostics.Debug.WriteLine($"[ApiClient] GetCharacters error: {ex.Message}");
            }
            return new List<CharacterModel>();
        }

        public async Task<CharacterModel?> CreateCharacterAsync(string name, string description, string gender, string? imageFilePath = null)
        {
            try
            {
                var endpoint = $"{_baseUrl}/api/characters";
                using var form = new MultipartFormDataContent();
                form.Add(new StringContent(name), "name");
                form.Add(new StringContent(description), "description");
                form.Add(new StringContent(gender), "gender");

                if (!string.IsNullOrEmpty(imageFilePath) && File.Exists(imageFilePath))
                {
                    var fileBytes = await File.ReadAllBytesAsync(imageFilePath);
                    var fileContent = new ByteArrayContent(fileBytes);
                    fileContent.Headers.ContentType = new System.Net.Http.Headers.MediaTypeHeaderValue("image/jpeg");
                    form.Add(fileContent, "images", Path.GetFileName(imageFilePath));
                }

                var response = await _httpClient.PostAsync(endpoint, form);
                if (response.IsSuccessStatusCode)
                {
                    return await response.Content.ReadFromJsonAsync<CharacterModel>();
                }
            }
            catch (Exception ex)
            {
                System.Diagnostics.Debug.WriteLine($"[ApiClient] CreateCharacter error: {ex.Message}");
            }
            return null;
        }

        public async Task<GenerationResponse?> GenerateShotAsync(GenerationRequest request)
        {
            try
            {
                var endpoint = $"{_baseUrl}/api/generate";
                var response = await _httpClient.PostAsJsonAsync(endpoint, request);
                var content = await response.Content.ReadAsStringAsync();

                var options = new JsonSerializerOptions { PropertyNameCaseInsensitive = true };
                var result = JsonSerializer.Deserialize<GenerationResponse>(content, options);

                // If video_url is a relative URL, prepend _baseUrl
                if (result != null && !string.IsNullOrEmpty(result.VideoUrl) && !result.VideoUrl.StartsWith("http"))
                {
                    result.VideoUrl = $"{_baseUrl}{result.VideoUrl}";
                }

                return result;
            }
            catch (Exception ex)
            {
                return new GenerationResponse
                {
                    Status = "error",
                    Error = ex.Message,
                    StatusMessage = $"Connection failed: {ex.Message}"
                };
            }
        }
    }
}

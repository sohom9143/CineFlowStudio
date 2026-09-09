using System;
using System.Collections.ObjectModel;
using System.Threading.Tasks;
using System.Windows.Input;
using CineFlowStudio.Desktop.Models;
using CineFlowStudio.Desktop.Services;

namespace CineFlowStudio.Desktop.ViewModels
{
    public class MainViewModel : ViewModelBase
    {
        private readonly IColabDiscoveryService _discoveryService;
        private readonly IColabApiClient _apiClient;

        // Bridge & Hardware State
        private string _colabUrl = string.Empty;
        private string _connectionStatus = "Discovering Colab GPU Bridge...";
        private bool _isOnline = false;
        private string _gpuDevice = "Tesla T4 (15GB VRAM)";

        // Generation Controls
        private string _prompt = "cinematic camera slow push in, 8k resolution, photorealistic cinematic lighting";
        private double _motionScale = 7.0;
        private double _cfgScale = 7.5;
        private int _inferenceSteps = 30;
        private string _selectedResolution = "480p";
        private string _selectedAspectRatio = "16:9";

        // Progress & Execution
        private bool _isGenerating = false;
        private string _generationStatusMessage = "Ready to direct scene.";
        private string _playbackUrl = string.Empty;
        private string _lastRenderLocalPath = string.Empty;

        // Character Management
        private CharacterModel? _selectedCharacter;

        public ObservableCollection<CharacterModel> Characters { get; } = new();
        public ObservableCollection<string> Resolutions { get; } = new() { "480p", "720p" };
        public ObservableCollection<string> AspectRatios { get; } = new() { "16:9", "9:16" };

        public MainViewModel(IColabDiscoveryService discoveryService, IColabApiClient apiClient)
        {
            _discoveryService = discoveryService;
            _apiClient = apiClient;

            RefreshBridgeCommand = new RelayCommand(async () => await AutoDiscoverColabAsync());
            GenerateShotCommand = new RelayCommand(async () => await ExecuteGenerateShotAsync(), () => !IsGenerating);
            Select16x9Command = new RelayCommand(() => SelectedAspectRatio = "16:9");
            Select9x16Command = new RelayCommand(() => SelectedAspectRatio = "9:16");
            Select480pCommand = new RelayCommand(() => SelectedResolution = "480p");
            Select720pCommand = new RelayCommand(() => SelectedResolution = "720p");

            // Kick off auto-discovery immediately on startup
            _ = AutoDiscoverColabAsync();
        }

        #region Properties

        public string ColabUrl
        {
            get => _colabUrl;
            set
            {
                if (SetProperty(ref _colabUrl, value))
                {
                    _apiClient.SetBaseUrl(value);
                }
            }
        }

        public string ConnectionStatus
        {
            get => _connectionStatus;
            set => SetProperty(ref _connectionStatus, value);
        }

        public bool IsOnline
        {
            get => _isOnline;
            set => SetProperty(ref _isOnline, value);
        }

        public string GpuDevice
        {
            get => _gpuDevice;
            set => SetProperty(ref _gpuDevice, value);
        }

        public string Prompt
        {
            get => _prompt;
            set => SetProperty(ref _prompt, value);
        }

        public double MotionScale
        {
            get => _motionScale;
            set => SetProperty(ref _motionScale, value);
        }

        public double CfgScale
        {
            get => _cfgScale;
            set => SetProperty(ref _cfgScale, value);
        }

        public int InferenceSteps
        {
            get => _inferenceSteps;
            set => SetProperty(ref _inferenceSteps, value);
        }

        public string SelectedResolution
        {
            get => _selectedResolution;
            set => SetProperty(ref _selectedResolution, value);
        }

        public string SelectedAspectRatio
        {
            get => _selectedAspectRatio;
            set => SetProperty(ref _selectedAspectRatio, value);
        }

        public bool IsGenerating
        {
            get => _isGenerating;
            set => SetProperty(ref _isGenerating, value);
        }

        public string GenerationStatusMessage
        {
            get => _generationStatusMessage;
            set => SetProperty(ref _generationStatusMessage, value);
        }

        public string PlaybackUrl
        {
            get => _playbackUrl;
            set => SetProperty(ref _playbackUrl, value);
        }

        public string LastRenderLocalPath
        {
            get => _lastRenderLocalPath;
            set => SetProperty(ref _lastRenderLocalPath, value);
        }

        public CharacterModel? SelectedCharacter
        {
            get => _selectedCharacter;
            set => SetProperty(ref _selectedCharacter, value);
        }

        #endregion

        #region Commands

        public ICommand RefreshBridgeCommand { get; }
        public ICommand GenerateShotCommand { get; }
        public ICommand Select16x9Command { get; }
        public ICommand Select9x16Command { get; }
        public ICommand Select480pCommand { get; }
        public ICommand Select720pCommand { get; }

        #endregion

        #region Actions

        public async Task AutoDiscoverColabAsync()
        {
            ConnectionStatus = "Auto-discovering Colab via Firebase RTDB...";
            IsOnline = false;

            var bridge = await _discoveryService.DiscoverActiveColabAsync();
            if (bridge != null && !string.IsNullOrEmpty(bridge.Url))
            {
                ColabUrl = bridge.Url;
                GpuDevice = $"{bridge.Device} ({bridge.Vram})";

                bool verified = await _discoveryService.VerifyConnectionAsync(bridge.Url);
                if (verified)
                {
                    IsOnline = true;
                    ConnectionStatus = $"🟢 Connected to {bridge.Device} ({bridge.Url})";
                    await LoadCharactersAsync();
                    return;
                }
            }

            // Fallback to local server if available
            bool localVerified = await _discoveryService.VerifyConnectionAsync("http://127.0.0.1:7860");
            if (localVerified)
            {
                ColabUrl = "http://127.0.0.1:7860";
                IsOnline = true;
                ConnectionStatus = "🟡 Connected to Local Python Engine (http://127.0.0.1:7860)";
                await LoadCharactersAsync();
                return;
            }

            IsOnline = false;
            ConnectionStatus = "🔴 Colab GPU Offline. Start Cell 6 in Colab to auto-connect.";
        }

        public async Task LoadCharactersAsync()
        {
            try
            {
                var chars = await _apiClient.GetCharactersAsync();
                Characters.Clear();
                foreach (var c in chars)
                {
                    Characters.Add(c);
                }

                if (Characters.Count > 0 && SelectedCharacter == null)
                {
                    SelectedCharacter = Characters[0];
                }
            }
            catch (Exception ex)
            {
                System.Diagnostics.Debug.WriteLine($"[MainViewModel] LoadCharacters failed: {ex.Message}");
            }
        }

        public async Task ExecuteGenerateShotAsync()
        {
            if (IsGenerating) return;

            IsGenerating = true;
            GenerationStatusMessage = $"🎬 Rendering shot in {SelectedResolution} ({SelectedAspectRatio}) on Colab T4 GPU...";

            var request = new GenerationRequest
            {
                CharacterId = SelectedCharacter?.Id ?? "",
                Prompt = Prompt,
                Motion = MotionScale,
                Cfg = CfgScale,
                Steps = InferenceSteps,
                AspectRatio = SelectedAspectRatio,
                Resolution = SelectedResolution,
                Model = "wan21",
            };

            try
            {
                var response = await _apiClient.GenerateShotAsync(request);
                if (response != null && response.IsSuccess)
                {
                    PlaybackUrl = response.EffectivePlaybackUrl;
                    LastRenderLocalPath = response.VideoPath;
                    GenerationStatusMessage = $"🎉 Master Shot Ready! ({SelectedResolution} | {SelectedAspectRatio})";
                }
                else
                {
                    GenerationStatusMessage = $"❌ Generation Failed: {response?.Error ?? "Unknown error"}";
                }
            }
            catch (Exception ex)
            {
                GenerationStatusMessage = $"❌ Execution Error: {ex.Message}";
            }
            finally
            {
                IsGenerating = false;
            }
        }

        #endregion
    }
}

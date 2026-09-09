using System;
using System.ComponentModel;
using System.Windows;
using CineFlowStudio.Desktop.Services;
using CineFlowStudio.Desktop.ViewModels;

namespace CineFlowStudio.Desktop
{
    public partial class MainWindow : Window
    {
        private readonly MainViewModel _viewModel;

        public MainWindow()
        {
            InitializeComponent();

            var discoveryService = new ColabDiscoveryService();
            var apiClient = new ColabApiClient();
            _viewModel = new MainViewModel(discoveryService, apiClient);
            DataContext = _viewModel;

            _viewModel.PropertyChanged += ViewModel_PropertyChanged;
        }

        private void ViewModel_PropertyChanged(object? sender, PropertyChangedEventArgs e)
        {
            if (e.PropertyName == nameof(MainViewModel.PlaybackUrl))
            {
                Dispatcher.Invoke(() =>
                {
                    if (!string.IsNullOrEmpty(_viewModel.PlaybackUrl))
                    {
                        try
                        {
                            MasterVideoPlayer.Source = new Uri(_viewModel.PlaybackUrl);
                            MasterVideoPlayer.Play();
                        }
                        catch (Exception ex)
                        {
                            System.Diagnostics.Debug.WriteLine($"Player error: {ex.Message}");
                        }
                    }
                });
            }
        }
    }
}

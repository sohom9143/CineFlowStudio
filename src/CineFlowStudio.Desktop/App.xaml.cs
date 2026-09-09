using System.Threading.Tasks;
using System.Windows;
using CineFlowStudio.Desktop.Services;

namespace CineFlowStudio.Desktop
{
    public partial class App : Application
    {
        protected override async void OnStartup(StartupEventArgs e)
        {
            base.OnStartup(e);

            // Automatically ensure local Python AI backend is running in the background
            await PythonBackendManager.EnsurePythonBackendRunningAsync();
        }

        protected override void OnExit(ExitEventArgs e)
        {
            // Clean up spawned Python background processes on application exit
            PythonBackendManager.StopPythonBackend();

            base.OnExit(e);
        }
    }
}

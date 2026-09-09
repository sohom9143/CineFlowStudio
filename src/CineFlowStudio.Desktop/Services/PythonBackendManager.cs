using System;
using System.Diagnostics;
using System.IO;
using System.Net.Sockets;
using System.Threading;
using System.Threading.Tasks;

namespace CineFlowStudio.Desktop.Services
{
    public class PythonBackendManager
    {
        private static Process? _pythonProcess;
        private static readonly object _lock = new();

        /// <summary>
        /// Finds the root project directory containing app.py.
        /// </summary>
        public static string? FindProjectRoot()
        {
            var dir = new DirectoryInfo(AppDomain.CurrentDomain.BaseDirectory);
            while (dir != null)
            {
                if (File.Exists(Path.Combine(dir.FullName, "app.py")))
                {
                    return dir.FullName;
                }
                dir = dir.Parent;
            }
            return null;
        }

        /// <summary>
        /// Checks if a TCP port is currently listening locally.
        /// </summary>
        public static bool IsPortOpen(string host, int port, int timeoutMs = 800)
        {
            try
            {
                using var client = new TcpClient();
                var result = client.BeginConnect(host, port, null, null);
                bool success = result.AsyncWaitHandle.WaitOne(timeoutMs);
                if (!success) return false;
                client.EndConnect(result);
                return true;
            }
            catch
            {
                return false;
            }
        }

        /// <summary>
        /// Ensures the local Python AI bridge backend is running.
        /// If not already active on port 7860, launches it in the background.
        /// </summary>
        public static async Task EnsurePythonBackendRunningAsync()
        {
            // If port 7860 is already open and responsive, no need to launch again
            if (IsPortOpen("127.0.0.1", 7860, 1000))
            {
                Debug.WriteLine("[PythonBackendManager] Local Python backend is already running on port 7860.");
                return;
            }

            lock (_lock)
            {
                if (_pythonProcess != null && !_pythonProcess.HasExited)
                {
                    return;
                }

                string? rootDir = FindProjectRoot();
                if (string.IsNullOrEmpty(rootDir))
                {
                    Debug.WriteLine("[PythonBackendManager] Could not locate project root containing app.py.");
                    return;
                }

                string appPyPath = Path.Combine(rootDir, "app.py");
                string configPath = Path.Combine(rootDir, "configs", "colab_t4_config.yaml");

                Debug.WriteLine($"[PythonBackendManager] Launching Python backend from {rootDir}...");

                var startInfo = new ProcessStartInfo
                {
                    FileName = "python",
                    Arguments = $"\"{appPyPath}\" --port 7860 --config \"{configPath}\"",
                    WorkingDirectory = rootDir,
                    UseShellExecute = false,
                    CreateNoWindow = true,
                    WindowStyle = ProcessWindowStyle.Hidden,
                    RedirectStandardOutput = false,
                    RedirectStandardError = false
                };

                try
                {
                    _pythonProcess = Process.Start(startInfo);
                    if (_pythonProcess != null)
                    {
                        Debug.WriteLine($"[PythonBackendManager] Spawned Python process with PID {_pythonProcess.Id}");
                    }
                }
                catch (Exception ex)
                {
                    Debug.WriteLine($"[PythonBackendManager] Failed to launch Python: {ex.Message}");
                }
            }

            // Wait up to 10 seconds for the backend server to spin up on port 7860
            for (int i = 0; i < 20; i++)
            {
                await Task.Delay(500);
                if (IsPortOpen("127.0.0.1", 7860, 500))
                {
                    Debug.WriteLine("[PythonBackendManager] Python backend is now alive and responding on port 7860.");
                    break;
                }
            }
        }

        /// <summary>
        /// Gracefully stops the background Python process when the desktop application closes.
        /// </summary>
        public static void StopPythonBackend()
        {
            lock (_lock)
            {
                if (_pythonProcess != null && !_pythonProcess.HasExited)
                {
                    try
                    {
                        Debug.WriteLine("[PythonBackendManager] Terminating background Python process...");
                        _pythonProcess.Kill(true); // Kill process tree
                        _pythonProcess.Dispose();
                    }
                    catch (Exception ex)
                    {
                        Debug.WriteLine($"[PythonBackendManager] Error closing Python process: {ex.Message}");
                    }
                    finally
                    {
                        _pythonProcess = null;
                    }
                }
            }
        }
    }
}

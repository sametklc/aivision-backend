import 'dart:async';
import 'dart:io';
import 'dart:math' as math;
import 'dart:ui';
import 'package:flutter/material.dart';
import 'package:lottie/lottie.dart';

/// Premium Full-Screen Generation Loading Experience
/// Cyberpunk / High-Tech / Cinematic Dark Mode
class GenerationLoadingScreen extends StatefulWidget {
  final String? inputImagePath;
  final String toolName;
  final Future<dynamic> generationFuture;
  final int estimatedSeconds;

  const GenerationLoadingScreen({
    super.key,
    this.inputImagePath,
    required this.toolName,
    required this.generationFuture,
    this.estimatedSeconds = 30,
  });

  @override
  State<GenerationLoadingScreen> createState() => _GenerationLoadingScreenState();
}

class _GenerationLoadingScreenState extends State<GenerationLoadingScreen>
    with TickerProviderStateMixin {
  // Progress simulation
  double _progress = 0.0;
  Timer? _progressTimer;
  Timer? _tipTimer;
  int _currentTipIndex = 0;
  bool _isComplete = false;
  dynamic _result;
  String? _error;

  // Animation controllers
  late AnimationController _glowController;
  late AnimationController _gradientController;
  late AnimationController _pulseController;

  // Cancel flag
  bool _isCancelled = false;

  // Status messages based on progress
  static const List<Map<String, dynamic>> _statusMessages = [
    {'range': 0.0, 'text': 'Initializing neural networks...', 'icon': Icons.memory},
    {'range': 0.15, 'text': 'Uploading assets to the cloud...', 'icon': Icons.cloud_upload},
    {'range': 0.25, 'text': 'Analyzing structure and composition...', 'icon': Icons.auto_fix_high},
    {'range': 0.40, 'text': 'Processing through AI layers...', 'icon': Icons.layers},
    {'range': 0.55, 'text': 'Dreaming up new pixels...', 'icon': Icons.auto_awesome},
    {'range': 0.70, 'text': 'Rendering high-quality output...', 'icon': Icons.hd},
    {'range': 0.85, 'text': 'Polishing the final details...', 'icon': Icons.brush},
    {'range': 0.95, 'text': 'Almost there, hang tight...', 'icon': Icons.hourglass_bottom},
  ];

  // Did you know tips
  static const List<String> _tips = [
    'Pro Tip: Better lighting in your photo means better AI results.',
    'Fun Fact: This AI model has billions of parameters.',
    'Did you know? AI processes images differently than humans.',
    'Pro Tip: High-resolution inputs produce sharper outputs.',
    'Fun Fact: Each generation creates unique, never-seen-before content.',
    'Pro Tip: Clear backgrounds help AI focus on the subject.',
    'Did you know? AI learns patterns from millions of images.',
    'Pro Tip: Consistent lighting reduces artifacts in results.',
  ];

  @override
  void initState() {
    super.initState();
    _initAnimations();
    _startProgressSimulation();
    _startTipRotation();
    _waitForGeneration();
  }

  void _initAnimations() {
    _glowController = AnimationController(
      duration: const Duration(milliseconds: 2000),
      vsync: this,
    )..repeat(reverse: true);

    _gradientController = AnimationController(
      duration: const Duration(seconds: 10),
      vsync: this,
    )..repeat();

    _pulseController = AnimationController(
      duration: const Duration(milliseconds: 1500),
      vsync: this,
    )..repeat(reverse: true);
  }

  void _startProgressSimulation() {
    // Calculate progress increment to reach 90% in estimated time
    final targetProgress = 0.90;
    final totalMs = widget.estimatedSeconds * 1000;
    final intervalMs = 100; // Update every 100ms
    final steps = totalMs / intervalMs;
    final increment = targetProgress / steps;

    _progressTimer = Timer.periodic(Duration(milliseconds: intervalMs), (timer) {
      if (!mounted) {
        timer.cancel();
        return;
      }

      setState(() {
        if (_progress < 0.90 && !_isComplete) {
          // Add some randomness for natural feel
          final randomFactor = 0.8 + (math.Random().nextDouble() * 0.4);
          _progress = math.min(0.90, _progress + (increment * randomFactor));
        }
      });
    });
  }

  void _startTipRotation() {
    _tipTimer = Timer.periodic(const Duration(seconds: 5), (timer) {
      if (!mounted) {
        timer.cancel();
        return;
      }
      setState(() {
        _currentTipIndex = (_currentTipIndex + 1) % _tips.length;
      });
    });
  }

  Future<void> _waitForGeneration() async {
    try {
      _result = await widget.generationFuture;
      // Don't proceed if cancelled
      if (_isCancelled || !mounted) return;

      setState(() {
        _isComplete = true;
        _progress = 1.0;
      });
      // Small delay to show 100% before popping
      await Future.delayed(const Duration(milliseconds: 500));
      if (mounted && !_isCancelled) {
        Navigator.of(context).pop(_result);
      }
    } catch (e) {
      // Don't show error if cancelled
      if (_isCancelled || !mounted) return;

      setState(() {
        _error = e.toString();
      });
      await Future.delayed(const Duration(seconds: 2));
      if (mounted && !_isCancelled) {
        Navigator.of(context).pop(null);
      }
    }
  }

  @override
  void dispose() {
    _progressTimer?.cancel();
    _tipTimer?.cancel();
    _glowController.dispose();
    _gradientController.dispose();
    _pulseController.dispose();
    super.dispose();
  }

  void _handleCancel() {
    setState(() {
      _isCancelled = true;
    });
    Navigator.of(context).pop(null);
  }

  String _getCurrentStatus() {
    for (int i = _statusMessages.length - 1; i >= 0; i--) {
      if (_progress >= _statusMessages[i]['range']) {
        return _statusMessages[i]['text'];
      }
    }
    return _statusMessages[0]['text'];
  }

  IconData _getCurrentIcon() {
    for (int i = _statusMessages.length - 1; i >= 0; i--) {
      if (_progress >= _statusMessages[i]['range']) {
        return _statusMessages[i]['icon'];
      }
    }
    return _statusMessages[0]['icon'];
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: Colors.black,
      body: Stack(
        children: [
          // Animated gradient background
          _buildAnimatedBackground(),

          // Main content
          SafeArea(
            child: Column(
              children: [
                const SizedBox(height: 40),

                // Tool name header
                _buildHeader(),

                const Spacer(flex: 1),

                // Input image preview
                if (widget.inputImagePath != null) _buildInputPreview(),

                const SizedBox(height: 32),

                // Central Lottie animation
                _buildCentralAnimation(),

                const SizedBox(height: 40),

                // Progress bar
                _buildProgressBar(),

                const SizedBox(height: 24),

                // Status text
                _buildStatusText(),

                const Spacer(flex: 2),

                // Did you know carousel
                _buildTipsCarousel(),

                const SizedBox(height: 40),
              ],
            ),
          ),

          // Cancel button (top-right)
          Positioned(
            top: MediaQuery.of(context).padding.top + 16,
            right: 16,
            child: GestureDetector(
              onTap: _handleCancel,
              child: Container(
                width: 44,
                height: 44,
                decoration: BoxDecoration(
                  shape: BoxShape.circle,
                  color: Colors.white.withOpacity(0.1),
                  border: Border.all(
                    color: Colors.white.withOpacity(0.2),
                    width: 1,
                  ),
                ),
                child: Icon(
                  Icons.close,
                  color: Colors.white.withOpacity(0.8),
                  size: 24,
                ),
              ),
            ),
          ),

          // Error overlay
          if (_error != null) _buildErrorOverlay(),
        ],
      ),
    );
  }

  Widget _buildAnimatedBackground() {
    return AnimatedBuilder(
      animation: _gradientController,
      builder: (context, child) {
        return Container(
          decoration: BoxDecoration(
            gradient: LinearGradient(
              begin: Alignment.topLeft,
              end: Alignment.bottomRight,
              colors: [
                const Color(0xFF0A0A0F),
                Color.lerp(
                  const Color(0xFF1A0A2E),
                  const Color(0xFF0A1A2E),
                  (math.sin(_gradientController.value * math.pi * 2) + 1) / 2,
                )!,
                const Color(0xFF0A0A0F),
              ],
              stops: [
                0.0,
                0.5 + (math.sin(_gradientController.value * math.pi * 2) * 0.2),
                1.0,
              ],
            ),
          ),
        );
      },
    );
  }

  Widget _buildHeader() {
    return Column(
      children: [
        Text(
          'GENERATING',
          style: TextStyle(
            color: Colors.white.withOpacity(0.5),
            fontSize: 12,
            fontWeight: FontWeight.w600,
            letterSpacing: 4,
          ),
        ),
        const SizedBox(height: 8),
        Text(
          widget.toolName.toUpperCase(),
          style: const TextStyle(
            color: Colors.white,
            fontSize: 24,
            fontWeight: FontWeight.bold,
            letterSpacing: 2,
          ),
        ),
      ],
    );
  }

  Widget _buildInputPreview() {
    return AnimatedBuilder(
      animation: _glowController,
      builder: (context, child) {
        final glowIntensity = 0.3 + (_glowController.value * 0.4);
        return Container(
          width: 120,
          height: 120,
          decoration: BoxDecoration(
            borderRadius: BorderRadius.circular(20),
            boxShadow: [
              BoxShadow(
                color: const Color(0xFF8B5CF6).withOpacity(glowIntensity),
                blurRadius: 30,
                spreadRadius: 5,
              ),
              BoxShadow(
                color: const Color(0xFF06B6D4).withOpacity(glowIntensity * 0.5),
                blurRadius: 50,
                spreadRadius: 10,
              ),
            ],
          ),
          child: ClipRRect(
            borderRadius: BorderRadius.circular(20),
            child: Stack(
              children: [
                // Image
                Image.file(
                  File(widget.inputImagePath!),
                  width: 120,
                  height: 120,
                  fit: BoxFit.cover,
                  errorBuilder: (context, error, stackTrace) {
                    return Container(
                      color: const Color(0xFF1A1A2E),
                      child: const Icon(
                        Icons.image,
                        color: Colors.white38,
                        size: 40,
                      ),
                    );
                  },
                ),
                // Scanning overlay effect
                Positioned.fill(
                  child: Container(
                    decoration: BoxDecoration(
                      gradient: LinearGradient(
                        begin: Alignment.topCenter,
                        end: Alignment.bottomCenter,
                        colors: [
                          Colors.transparent,
                          const Color(0xFF8B5CF6).withOpacity(0.1),
                          Colors.transparent,
                        ],
                        stops: [
                          (_glowController.value - 0.2).clamp(0.0, 1.0),
                          _glowController.value,
                          (_glowController.value + 0.2).clamp(0.0, 1.0),
                        ],
                      ),
                    ),
                  ),
                ),
                // Border
                Positioned.fill(
                  child: Container(
                    decoration: BoxDecoration(
                      borderRadius: BorderRadius.circular(20),
                      border: Border.all(
                        color: const Color(0xFF8B5CF6).withOpacity(0.5 + glowIntensity * 0.5),
                        width: 2,
                      ),
                    ),
                  ),
                ),
              ],
            ),
          ),
        );
      },
    );
  }

  Widget _buildCentralAnimation() {
    return SizedBox(
      width: 200,
      height: 200,
      child: Stack(
        alignment: Alignment.center,
        children: [
          // Outer rotating ring
          AnimatedBuilder(
            animation: _gradientController,
            builder: (context, child) {
              return Transform.rotate(
                angle: _gradientController.value * math.pi * 2,
                child: Container(
                  width: 200,
                  height: 200,
                  decoration: BoxDecoration(
                    shape: BoxShape.circle,
                    gradient: SweepGradient(
                      colors: [
                        const Color(0xFF8B5CF6).withOpacity(0.0),
                        const Color(0xFF8B5CF6).withOpacity(0.5),
                        const Color(0xFF06B6D4).withOpacity(0.5),
                        const Color(0xFF8B5CF6).withOpacity(0.0),
                      ],
                    ),
                  ),
                ),
              );
            },
          ),
          // Inner glow circle
          AnimatedBuilder(
            animation: _pulseController,
            builder: (context, child) {
              return Container(
                width: 160 + (_pulseController.value * 10),
                height: 160 + (_pulseController.value * 10),
                decoration: BoxDecoration(
                  shape: BoxShape.circle,
                  boxShadow: [
                    BoxShadow(
                      color: const Color(0xFF8B5CF6).withOpacity(0.3),
                      blurRadius: 40,
                      spreadRadius: 10,
                    ),
                  ],
                ),
              );
            },
          ),
          // Lottie animation
          ClipOval(
            child: Container(
              width: 150,
              height: 150,
              decoration: BoxDecoration(
                color: const Color(0xFF0A0A0F).withOpacity(0.8),
                shape: BoxShape.circle,
              ),
              child: Lottie.network(
                'https://assets5.lottiefiles.com/packages/lf20_ofa3xwo7.json',
                width: 150,
                height: 150,
                fit: BoxFit.contain,
                errorBuilder: (context, error, stackTrace) {
                  // Fallback to custom animation
                  return _buildFallbackAnimation();
                },
              ),
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildFallbackAnimation() {
    return AnimatedBuilder(
      animation: Listenable.merge([_pulseController, _gradientController]),
      builder: (context, child) {
        return CustomPaint(
          size: const Size(150, 150),
          painter: _AIBrainPainter(
            pulseValue: _pulseController.value,
            rotateValue: _gradientController.value,
          ),
        );
      },
    );
  }

  Widget _buildProgressBar() {
    return Padding(
      padding: const EdgeInsets.symmetric(horizontal: 48),
      child: Column(
        children: [
          // Progress percentage
          Row(
            mainAxisAlignment: MainAxisAlignment.spaceBetween,
            children: [
              Text(
                'Progress',
                style: TextStyle(
                  color: Colors.white.withOpacity(0.6),
                  fontSize: 12,
                  fontWeight: FontWeight.w500,
                ),
              ),
              Text(
                '${(_progress * 100).toInt()}%',
                style: const TextStyle(
                  color: Color(0xFF8B5CF6),
                  fontSize: 14,
                  fontWeight: FontWeight.bold,
                ),
              ),
            ],
          ),
          const SizedBox(height: 8),
          // Progress bar
          Container(
            height: 6,
            decoration: BoxDecoration(
              borderRadius: BorderRadius.circular(3),
              color: Colors.white.withOpacity(0.1),
            ),
            child: Stack(
              children: [
                // Animated progress fill
                AnimatedBuilder(
                  animation: _glowController,
                  builder: (context, child) {
                    return FractionallySizedBox(
                      alignment: Alignment.centerLeft,
                      widthFactor: _progress,
                      child: Container(
                        decoration: BoxDecoration(
                          borderRadius: BorderRadius.circular(3),
                          gradient: const LinearGradient(
                            colors: [
                              Color(0xFF8B5CF6),
                              Color(0xFF06B6D4),
                            ],
                          ),
                          boxShadow: [
                            BoxShadow(
                              color: const Color(0xFF8B5CF6).withOpacity(0.5 + _glowController.value * 0.3),
                              blurRadius: 10,
                              spreadRadius: 1,
                            ),
                          ],
                        ),
                      ),
                    );
                  },
                ),
                // Shimmer effect
                if (_progress > 0 && _progress < 1)
                  AnimatedBuilder(
                    animation: _glowController,
                    builder: (context, child) {
                      return Positioned(
                        left: (_progress * MediaQuery.of(context).size.width * 0.8) - 20,
                        child: Container(
                          width: 40,
                          height: 6,
                          decoration: BoxDecoration(
                            gradient: LinearGradient(
                              colors: [
                                Colors.white.withOpacity(0),
                                Colors.white.withOpacity(_glowController.value * 0.5),
                                Colors.white.withOpacity(0),
                              ],
                            ),
                          ),
                        ),
                      );
                    },
                  ),
              ],
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildStatusText() {
    return AnimatedSwitcher(
      duration: const Duration(milliseconds: 500),
      transitionBuilder: (child, animation) {
        return FadeTransition(
          opacity: animation,
          child: SlideTransition(
            position: Tween<Offset>(
              begin: const Offset(0, 0.3),
              end: Offset.zero,
            ).animate(animation),
            child: child,
          ),
        );
      },
      child: Row(
        key: ValueKey<String>(_getCurrentStatus()),
        mainAxisAlignment: MainAxisAlignment.center,
        children: [
          AnimatedBuilder(
            animation: _pulseController,
            builder: (context, child) {
              return Icon(
                _getCurrentIcon(),
                color: Color.lerp(
                  const Color(0xFF8B5CF6),
                  const Color(0xFF06B6D4),
                  _pulseController.value,
                ),
                size: 20,
              );
            },
          ),
          const SizedBox(width: 12),
          Text(
            _getCurrentStatus(),
            style: TextStyle(
              color: Colors.white.withOpacity(0.9),
              fontSize: 16,
              fontWeight: FontWeight.w500,
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildTipsCarousel() {
    return Container(
      margin: const EdgeInsets.symmetric(horizontal: 32),
      padding: const EdgeInsets.all(20),
      decoration: BoxDecoration(
        borderRadius: BorderRadius.circular(16),
        color: Colors.white.withOpacity(0.05),
        border: Border.all(
          color: Colors.white.withOpacity(0.1),
          width: 1,
        ),
      ),
      child: Column(
        children: [
          Row(
            children: [
              Container(
                padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 4),
                decoration: BoxDecoration(
                  borderRadius: BorderRadius.circular(6),
                  gradient: LinearGradient(
                    colors: [
                      const Color(0xFF8B5CF6).withOpacity(0.3),
                      const Color(0xFF06B6D4).withOpacity(0.3),
                    ],
                  ),
                ),
                child: const Text(
                  'DID YOU KNOW?',
                  style: TextStyle(
                    color: Colors.white,
                    fontSize: 10,
                    fontWeight: FontWeight.bold,
                    letterSpacing: 1,
                  ),
                ),
              ),
            ],
          ),
          const SizedBox(height: 12),
          AnimatedSwitcher(
            duration: const Duration(milliseconds: 600),
            transitionBuilder: (child, animation) {
              return FadeTransition(
                opacity: animation,
                child: child,
              );
            },
            child: Text(
              _tips[_currentTipIndex],
              key: ValueKey<int>(_currentTipIndex),
              textAlign: TextAlign.center,
              style: TextStyle(
                color: Colors.white.withOpacity(0.7),
                fontSize: 14,
                height: 1.5,
              ),
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildErrorOverlay() {
    return Container(
      color: Colors.black.withOpacity(0.8),
      child: Center(
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            const Icon(
              Icons.error_outline,
              color: Color(0xFFEF4444),
              size: 64,
            ),
            const SizedBox(height: 16),
            const Text(
              'Generation Failed',
              style: TextStyle(
                color: Colors.white,
                fontSize: 20,
                fontWeight: FontWeight.bold,
              ),
            ),
            const SizedBox(height: 8),
            Padding(
              padding: const EdgeInsets.symmetric(horizontal: 48),
              child: Text(
                _error ?? 'Unknown error',
                textAlign: TextAlign.center,
                style: TextStyle(
                  color: Colors.white.withOpacity(0.6),
                  fontSize: 14,
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }
}

/// Custom painter for fallback AI brain animation
class _AIBrainPainter extends CustomPainter {
  final double pulseValue;
  final double rotateValue;

  _AIBrainPainter({
    required this.pulseValue,
    required this.rotateValue,
  });

  @override
  void paint(Canvas canvas, Size size) {
    final center = Offset(size.width / 2, size.height / 2);
    final maxRadius = size.width / 2;

    // Draw neural network nodes
    final nodePaint = Paint()
      ..color = const Color(0xFF8B5CF6).withOpacity(0.8);

    final linePaint = Paint()
      ..color = const Color(0xFF06B6D4).withOpacity(0.3)
      ..strokeWidth = 1;

    // Central node
    canvas.drawCircle(center, 8 + pulseValue * 4, nodePaint);

    // Outer nodes
    for (int i = 0; i < 8; i++) {
      final angle = (i / 8) * math.pi * 2 + rotateValue * math.pi * 2;
      final radius = maxRadius * 0.7;
      final nodePos = center + Offset(
        math.cos(angle) * radius,
        math.sin(angle) * radius,
      );

      // Connection line
      canvas.drawLine(center, nodePos, linePaint);

      // Node
      canvas.drawCircle(nodePos, 4 + pulseValue * 2, nodePaint);
    }

    // Inner ring nodes
    for (int i = 0; i < 6; i++) {
      final angle = (i / 6) * math.pi * 2 - rotateValue * math.pi;
      final radius = maxRadius * 0.4;
      final nodePos = center + Offset(
        math.cos(angle) * radius,
        math.sin(angle) * radius,
      );

      canvas.drawCircle(nodePos, 3 + pulseValue * 1.5, nodePaint..color = const Color(0xFF06B6D4).withOpacity(0.6));
    }
  }

  @override
  bool shouldRepaint(_AIBrainPainter oldDelegate) {
    return pulseValue != oldDelegate.pulseValue ||
        rotateValue != oldDelegate.rotateValue;
  }
}

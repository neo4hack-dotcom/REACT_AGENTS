#!/usr/bin/env node

import { spawn, spawnSync } from 'node:child_process';
import { existsSync } from 'node:fs';
import path from 'node:path';
import process from 'node:process';

const rootDir = process.cwd();
const isWindows = process.platform === 'win32';
const npmCmd = isWindows ? 'npm.cmd' : 'npm';

function detectPython() {
  const candidates = isWindows
    ? [
        { cmd: 'py', prefix: ['-3'] },
        { cmd: 'python', prefix: [] },
        { cmd: 'python3', prefix: [] },
      ]
    : [
        { cmd: 'python3', prefix: [] },
        { cmd: 'python', prefix: [] },
      ];

  for (const candidate of candidates) {
    const probe = spawnSync(
      candidate.cmd,
      [...candidate.prefix, '-c', 'import sys; print(sys.version)'],
      { stdio: 'ignore' }
    );

    if (probe.status === 0) {
      return candidate;
    }
  }

  return null;
}

function runStep(title, cmd, args) {
  return new Promise((resolve, reject) => {
    console.log(`\n==> ${title}`);
    const child = spawn(cmd, args, {
      cwd: rootDir,
      stdio: 'inherit',
      env: process.env,
    });

    child.on('error', reject);
    child.on('exit', (code) => {
      if (code === 0) {
        resolve();
      } else {
        reject(new Error(`${title} failed with code ${code}`));
      }
    });
  });
}

async function main() {
  const python = detectPython();
  if (!python) {
    console.error('No Python 3 interpreter was found. Install Python 3.11+ and retry.');
    process.exit(1);
  }

  const venvDir = path.join(rootDir, '.venv');
  const venvPython = isWindows
    ? path.join(venvDir, 'Scripts', 'python.exe')
    : path.join(venvDir, 'bin', 'python');

  try {
    await runStep('Installing frontend dependencies (npm)', npmCmd, ['install']);

    if (!existsSync(venvPython)) {
      await runStep('Creating Python virtual environment (.venv)', python.cmd, [...python.prefix, '-m', 'venv', '.venv']);
    }

    await runStep('Upgrading pip in virtual environment', venvPython, ['-m', 'pip', 'install', '--upgrade', 'pip']);
    await runStep('Installing backend dependencies (pip)', venvPython, ['-m', 'pip', 'install', '-r', 'backend/requirements.txt']);
  } catch (error) {
    console.error(`\nSetup failed: ${error.message}`);
    process.exit(1);
  }

  console.log('\n==> Starting application');
  console.log('    UI:      http://localhost:5173');
  console.log('    Backend: http://localhost:3000');

  const backend = spawn(
    venvPython,
    ['-m', 'uvicorn', 'backend.main:app', '--reload', '--host', '0.0.0.0', '--port', '3000'],
    {
      cwd: rootDir,
      stdio: 'inherit',
      env: process.env,
    }
  );

  const frontend = spawn(npmCmd, ['run', 'dev:frontend'], {
    cwd: rootDir,
    stdio: 'inherit',
    env: process.env,
  });

  let shuttingDown = false;

  const shutdown = (signal = 'SIGTERM') => {
    if (shuttingDown) return;
    shuttingDown = true;

    backend.kill(signal);
    frontend.kill(signal);

    setTimeout(() => {
      if (!backend.killed) backend.kill('SIGKILL');
      if (!frontend.killed) frontend.kill('SIGKILL');
    }, 2000);
  };

  const handleExit = (name, code) => {
    if (!shuttingDown) {
      console.error(`\n${name} exited unexpectedly with code ${code ?? 'unknown'}.`);
      shutdown('SIGTERM');
      process.exit(code ?? 1);
    }
  };

  backend.on('exit', (code) => handleExit('Backend', code));
  frontend.on('exit', (code) => handleExit('Frontend', code));

  process.on('SIGINT', () => {
    shutdown('SIGINT');
    process.exit(0);
  });

  process.on('SIGTERM', () => {
    shutdown('SIGTERM');
    process.exit(0);
  });
}

main();

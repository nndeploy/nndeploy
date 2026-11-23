"""Exercise setup metadata without installing dependencies or native libraries."""
import contextlib
import io
import os
from pathlib import Path
import runpy
import tempfile
import unittest
from unittest.mock import patch

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name
from setuptools.dist import Distribution


SETUP = Path(__file__).resolve().parents[2] / 'python' / 'setup.py'


class PackageDependenciesTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.build = self.root / 'python'
        (self.build / 'nndeploy').mkdir(parents=True)
        # get_package_data requires a native library; all inspection/patching
        # subprocesses are mocked and only this disposable file is touched.
        (self.build / 'nndeploy' / ('stub.pyd' if os.name == 'nt' else 'stub.so')).touch()

    def metadata(self, override=None, algorithm=None):
        previous = os.getcwd()
        try:
            os.chdir(self.build)
            with patch.dict(os.environ), patch('setuptools.setup') as setup, \
                    patch('subprocess.run'), contextlib.redirect_stdout(io.StringIO()):
                os.environ.pop('NNDEPLOY_REQUIREMENTS_OVERRIDE', None)
                os.environ.pop('NNDEPLOY_DIT_ALGORITHM', None)
                if override is not None:
                    os.environ['NNDEPLOY_REQUIREMENTS_OVERRIDE'] = str(override)
                if algorithm is not None:
                    os.environ['NNDEPLOY_DIT_ALGORITHM'] = algorithm
                runpy.run_path(str(SETUP), run_name='__main__')
            return setup.call_args.kwargs
        finally:
            os.chdir(previous)

    def override(self, content):
        filename = self.root / 'algorithm.txt'
        filename.write_text(content, encoding='utf-8')
        return filename

    def test_default_install_and_requirements_file_exclude_dit_dependencies(self):
        metadata = self.metadata()
        requirements = metadata['install_requires']
        by_name = {canonicalize_name(Requirement(item).name): item for item in requirements}
        self.assertEqual(len(by_name), len(requirements))
        root_requirements = {
            line for line in (SETUP.parents[1] / 'requirements.txt').read_text().splitlines()
            if line and not line.startswith('#')
        }
        self.assertTrue(set(requirements).issubset(root_requirements))
        root_names = {canonicalize_name(Requirement(item).name) for item in root_requirements}
        self.assertTrue({'triton', 'flash-attn'}.isdisjoint(root_names))
        extras = metadata['extras_require']
        for item in (
            'torch==2.6.0', 'torchvision==0.21.0', 'diffusers==0.40.0',
            'transformers==5.3.0', 'accelerate==1.13.0', 'peft>=0.12',
            'einops', 'sentencepiece', 'safetensors>=0.8', 'rich',
            'imageio-ffmpeg', 'ffmpeg-python', 'decord', 'moviepy',
            'pycocotools', 'pandas', 'scipy', 'ftfy==6.3.1',
        ):
            self.assertIn(item, extras['e_dit'])
            self.assertNotIn(canonicalize_name(Requirement(item).name), root_names)
        for item in (
            'av>=16.1,<20',
            'mcp>=1.30,<2', 'pymysql>=1.1,<2', 'psycopg[binary]>=3.2,<4',
            'redis>=5,<8', 'boto3>=1.34,<2', 'cos-python-sdk-v5>=1.9,<2',
            'alibabacloud-oss-v2>=1.4,<2', 'esdk-obs-python>=3.25,<4',
        ):
            self.assertIn(item, requirements)
        self.assertEqual(len(extras), 10)
        self.assertTrue(all(not items for name, items in extras.items() if name.startswith('server-')))

    def test_replace_versions_add_packages_and_emit_updated_metadata(self):
        filename = self.override(
            '# Algorithm-specific versions\n\n'
            'torch==2.7.0 # Replace the pin, do not intersect it\n'
            'torchvision==0.22.0\n'
            'OpenCV_Python==4.11.0.86\n'
            'psycopg[binary]==3.2.6\n'
            'new-algorithm-lib>=1\n'
        )
        metadata = self.metadata(filename)
        requirements = metadata['install_requires']
        self.assertIn('torch==2.7.0', requirements)
        self.assertNotIn('torch==2.6.0', requirements)
        self.assertNotIn('torchvision==0.21.0', requirements)
        self.assertNotIn('opencv-python>=4.10.0', requirements)
        self.assertIn('OpenCV_Python==4.11.0.86', requirements)
        self.assertIn('new-algorithm-lib>=1', requirements)
        self.assertIn('psycopg[binary]==3.2.6', requirements)
        self.assertIn('fastapi>=0.104.0', requirements)
        self.assertIn('diffusers==0.40.0', requirements)
        # Check serialized distribution metadata, including absence of extras
        # that would silently reintroduce the original pins.
        distribution = Distribution({
            'name': metadata['name'], 'version': metadata['version'],
            'install_requires': requirements, 'extras_require': metadata['extras_require'],
        })
        output = io.StringIO()
        distribution.metadata.write_pkg_file(output)
        self.assertIn('Requires-Dist: torch==2.7.0\n', output.getvalue())
        self.assertNotIn('torch==2.6.0', output.getvalue())

    def test_relative_path_uses_setup_directory_and_reinstall_resets_defaults(self):
        filename = self.override('torch==2.7.0\n')
        relative = os.path.relpath(filename, SETUP.parent)
        self.assertIn('torch==2.7.0', self.metadata(relative)['install_requires'])
        self.assertIn('torch==2.6.0', self.metadata()['extras_require']['e_dit'])

    def test_markers_and_url_fragments_are_preserved(self):
        filename = self.override(
            'torch==2.7.0; sys_platform == "linux"\n'
            'torch==2.6.0; sys_platform != "linux"\n'
            'new-lib @ https://example.com/new_lib.whl#sha256=abc\n'
        )
        requirements = self.metadata(filename)['install_requires']
        torch_requirements = [Requirement(item) for item in requirements
                              if Requirement(item).name == 'torch']
        self.assertEqual(len(torch_requirements), 2)
        self.assertTrue(all(item.marker for item in torch_requirements))
        self.assertTrue(any('#sha256=abc' in item for item in requirements))

    def test_invalid_or_missing_override_fails_instead_of_using_defaults(self):
        for content in ('--extra-index-url', '--unexpected-option value', '-r base.txt', 'torch=>2'):
            with self.subTest(content=content), self.assertRaisesRegex(ValueError, 'algorithm.txt:1'):
                self.metadata(self.override(content))
        with self.assertRaises(FileNotFoundError):
            self.metadata(self.root / 'missing.txt')

    def test_empty_algorithm_profile_uses_default_dit_versions(self):
        defaults = self.metadata()
        profile = self.metadata(self.override('# No overrides\n'))
        self.assertEqual(defaults['install_requires'] + defaults['extras_require']['e_dit'],
                         profile['install_requires'])

    def test_algorithm_selection_reads_real_profile_and_replaces_defaults(self):
        metadata = self.metadata(algorithm='eraserdit')
        requirements = metadata['install_requires']
        for item in ('torch==2.6.0+cu126', 'torchvision==0.21.0+cu126',
                     'numpy==2.2.6', 'pillow==10.4.0', 'flash-attn==2.8.3'):
            self.assertIn(item, requirements)
        for item in ('torch==2.6.0', 'numpy', 'Pillow', 'fastapi>=0.104.0'):
            self.assertNotIn(item, requirements)
        self.assertIn('mcp>=1.30,<2', requirements)
        self.assertFalse(any(item.startswith('--') for item in requirements))
        self.assertIn('e_dit/algorithms/*/requirements.txt', metadata['package_data']['nndeploy'])
        # A subsequent plain installation must not inherit the selected profile.
        self.assertNotIn('torch==2.6.0+cu126', self.metadata()['extras_require']['e_dit'])

    def test_algorithm_selection_rejects_missing_invalid_or_ambiguous_profiles(self):
        with self.assertRaisesRegex(FileNotFoundError, 'Algorithm requirements not found'):
            self.metadata(algorithm='no_such_algorithm')
        for name in ('../eraserdit', '/eraserdit', 'eraserdit,wan'):
            with self.subTest(name=name), self.assertRaises(ValueError):
                self.metadata(algorithm=name)
        with self.assertRaisesRegex(ValueError, 'Set only one'):
            self.metadata(override=self.override('numpy==2.2.6'), algorithm='eraserdit')

    def test_index_options_are_excluded_from_package_metadata(self):
        filename = self.override(
            '--extra-index-url https://example.com/simple?key=value\n'
            '--index-url=https://example.com/simple\n'
            '--find-links /tmp/wheels\n'
            '--trusted-host example.com\n'
            '--no-index\n'
            'torch==2.7.0\n'
        )
        requirements = self.metadata(filename)['install_requires']
        self.assertIn('torch==2.7.0', requirements)
        self.assertFalse(any(item.startswith('--') for item in requirements))

    def test_installer_passes_profile_indexes_and_metadata_selection_to_same_pip(self):
        installer = runpy.run_path(str(SETUP.with_name('install.py')))
        with patch.dict(os.environ), patch('subprocess.run') as run, \
                contextlib.redirect_stdout(io.StringIO()):
            os.environ.pop('NNDEPLOY_REQUIREMENTS_OVERRIDE', None)
            run.return_value.returncode = 7
            result = installer['main'](['--algorithm', 'eraserdit', '--editable', '--dry-run'])
        self.assertEqual(result, 7)
        command = run.call_args.args[0]
        self.assertEqual(command[1:4], ['-m', 'pip', 'install'])
        profile = SETUP.parent / 'nndeploy/e_dit/algorithms/eraserdit/requirements.txt'
        self.assertEqual(command[command.index('-r') + 1], str(profile))
        self.assertEqual(command[command.index('--editable') + 1], str(SETUP.parent))
        self.assertIn('--dry-run', command)
        self.assertEqual(run.call_args.kwargs['env']['NNDEPLOY_DIT_ALGORITHM'], 'eraserdit')
        self.assertEqual(run.call_args.kwargs['cwd'], SETUP.parent)

    def test_installer_rejects_invalid_profile_before_running_pip(self):
        installer = runpy.run_path(str(SETUP.with_name('install.py')))
        for name in ('../eraserdit', 'no_such_algorithm'):
            with self.subTest(name=name), patch('subprocess.run') as run, \
                    contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                installer['main'](['--algorithm', name])
            run.assert_not_called()

    def test_installer_resolves_file_relative_to_script_from_another_directory(self):
        installer = runpy.run_path(str(SETUP.with_name('install.py')))
        relative = 'nndeploy/e_dit/algorithms/eraserdit/requirements.txt'
        previous = os.getcwd()
        try:
            os.chdir(self.root)
            with patch.dict(os.environ, {'NNDEPLOY_DIT_ALGORITHM': 'wan'}), \
                    patch('subprocess.run') as run, contextlib.redirect_stdout(io.StringIO()):
                run.return_value.returncode = 0
                self.assertEqual(installer['main']([relative, '--editable']), 0)
        finally:
            os.chdir(previous)
        expected = str(SETUP.parent / relative)
        command = run.call_args.args[0]
        self.assertEqual(command[command.index('-r') + 1], expected)
        environment = run.call_args.kwargs['env']
        self.assertEqual(environment['NNDEPLOY_REQUIREMENTS_OVERRIDE'], expected)
        self.assertNotIn('NNDEPLOY_DIT_ALGORITHM', environment)

    def test_installer_accepts_absolute_requirements_path(self):
        installer = runpy.run_path(str(SETUP.with_name('install.py')))
        filename = self.override('torch==2.7.0\n')
        with patch('subprocess.run') as run, contextlib.redirect_stdout(io.StringIO()):
            installer['main']([str(filename), '--dry-run'])
        self.assertEqual(run.call_args.kwargs['env']['NNDEPLOY_REQUIREMENTS_OVERRIDE'], str(filename))
        self.assertIn('--dry-run', run.call_args.args[0])

    def test_installer_rejects_missing_or_ambiguous_file_selection(self):
        installer = runpy.run_path(str(SETUP.with_name('install.py')))
        for args in ([], ['missing.txt'], ['missing.txt', '--algorithm', 'eraserdit']):
            with self.subTest(args=args), patch('subprocess.run') as run, \
                    contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                installer['main'](args)
            run.assert_not_called()

    def test_installer_accepts_absolute_and_relative_directories(self):
        installer = runpy.run_path(str(SETUP.with_name('install.py')))
        directory = self.root / 'algorithm'
        directory.mkdir()
        filename = directory / 'requirements.txt'
        filename.write_text('torch==2.7.0\n', encoding='utf-8')
        for argument in (str(directory), os.path.relpath(directory, SETUP.parent)):
            with self.subTest(argument=argument), patch('subprocess.run') as run, \
                    contextlib.redirect_stdout(io.StringIO()):
                installer['main']([argument])
            command = run.call_args.args[0]
            self.assertEqual(command[command.index('-r') + 1], str(filename))
            self.assertEqual(run.call_args.kwargs['env']['NNDEPLOY_REQUIREMENTS_OVERRIDE'], str(filename))

    def test_installer_rejects_directory_without_requirements(self):
        installer = runpy.run_path(str(SETUP.with_name('install.py')))
        with patch('subprocess.run') as run, contextlib.redirect_stderr(io.StringIO()), \
                self.assertRaises(SystemExit):
            installer['main']([str(self.root)])
        run.assert_not_called()


if __name__ == '__main__':
    unittest.main()
